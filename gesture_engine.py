from __future__ import annotations
import json, time
from collections import deque, Counter
from pathlib import Path
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import confusion_matrix, accuracy_score
from kinematics import kinematic_summary, movement_metrics
from neural_gru import GRUGestureModel

class GestureEngine:
    def __init__(self, data_dir='data', model_dir='models', window=18):
        self.data_dir=Path(data_dir); self.model_dir=Path(model_dir)
        self.data_dir.mkdir(parents=True,exist_ok=True); self.model_dir.mkdir(parents=True,exist_ok=True)
        self.samples_path=self.data_dir/'gesture_samples.npz'; self.seq_path=self.data_dir/'gesture_sequences.npz'; self.meta_path=self.data_dir/'profile.json'; self.model_path=self.model_dir/'gesture_model.joblib'
        self.window=window; self.buffer=deque(maxlen=window); self.recording=None
        self.samples=[]; self.seq_samples=[]; self.labels=[]; self.phrases={}; self.emergency_labels=set(); self.model=None; self.metrics={}; self.history=[]
        self.gru=GRUGestureModel(model_dir=self.model_dir); self.last_fusion=None
        self.strength_baselines={}; self.drift_history={}; self.last_drift=None
        self.pred_history=deque(maxlen=8); self.cooldown_until=0; self.pending_prefix=None; self.pending_until=0
        # Safety gate: the recognizer must observe a stable REST state before a
        # non-rest gesture is allowed to trigger speech. After every trigger it
        # disarms again until the user returns to REST.
        self.armed=False
        self.prefixes={'WANT':'I want','NEED':'I need','CALL':'Please call','PLEASE':'Please'}
        self.rest_labels={'REST','NONE','NEUTRAL'}
        self._load()

    def _load(self):
        if self.samples_path.exists():
            z=np.load(self.samples_path,allow_pickle=True); self.samples=list(z['X']); self.labels=list(z['y'])
        if self.seq_path.exists():
            z=np.load(self.seq_path,allow_pickle=True); self.seq_samples=list(z['Xseq'])
        if self.meta_path.exists():
            meta=json.loads(self.meta_path.read_text()); self.phrases=meta.get('phrases',{}); self.history=meta.get('history',[])[-50:]
            self.emergency_labels=set(meta.get('emergency_labels',[])); self.metrics=meta.get('metrics',{})
            self.strength_baselines=meta.get('strength_baselines',{})
            self.drift_history=meta.get('drift_history',{})
        if self.model_path.exists():
            try:self.model=joblib.load(self.model_path)
            except Exception:self.model=None

    def _save(self):
        if self.samples: np.savez_compressed(self.samples_path,X=np.asarray(self.samples),y=np.asarray(self.labels,dtype=object))
        if self.seq_samples: np.savez_compressed(self.seq_path,Xseq=np.asarray(self.seq_samples,dtype=np.float32))
        self.meta_path.write_text(json.dumps({'phrases':self.phrases,'history':self.history[-50:],'emergency_labels':sorted(self.emergency_labels),'metrics':self.metrics,'strength_baselines':self.strength_baselines,'drift_history':self.drift_history},indent=2))

    @staticmethod
    def _apply_dead_zones(seq, body_xy_deadzone=0.0035, hand_deadzone=0.0045):
        """Suppress tiny frame-to-frame landmark jitter without removing intentional motion.

        Feature layout is body 9x4 (36 values) followed by hand landmarks when present.
        Dead-zones are body-normalized and intentionally small.
        """
        a=np.asarray(seq,dtype=np.float32).copy()
        if a.ndim!=2 or len(a)<2: return a
        # Body x/y only; preserve z/visibility.
        if a.shape[1]>=36:
            body=a[:,:36].reshape(len(a),9,4)
            for t in range(1,len(body)):
                d=body[t,:,:2]-body[t-1,:,:2]
                mask=np.abs(d)<body_xy_deadzone
                body[t,:,:2][mask]=body[t-1,:,:2][mask]
            a[:,:36]=body.reshape(len(a),36)
        # Hand x/y/z triplets, if present. Visibility is not part of hand layout.
        if a.shape[1]>36:
            hand=a[:,36:]
            for t in range(1,len(hand)):
                d=hand[t]-hand[t-1]
                hand[t][np.abs(d)<hand_deadzone]=hand[t-1][np.abs(d)<hand_deadzone]
            a[:,36:]=hand
        return a

    @staticmethod
    def summarize(seq):
        a=GestureEngine._apply_dead_zones(seq)
        if a.ndim!=2 or len(a)<4: raise ValueError('Not enough frames')
        target=18; old=np.linspace(0,1,len(a)); new=np.linspace(0,1,target)
        interp=np.vstack([np.interp(new,old,a[:,i]) for i in range(a.shape[1])]).T; d=np.diff(interp,axis=0)
        stats=np.concatenate([interp.mean(0),interp.std(0),interp.min(0),interp.max(0),interp[-1]-interp[0],np.abs(d).mean(0),np.abs(d).max(0)])
        traj=interp[[0,4,8,13,17]].reshape(-1)
        return np.concatenate([stats,traj]).astype(np.float32)


    @staticmethod
    def movement_strength(seq):
        """Body-normalized upper-body path length used as a gesture-strength proxy.

        This is not a medical measurement. It is only used to compare one user's
        current gesture with that same user's recorded examples.
        """
        a=np.asarray(seq,dtype=np.float32)
        if a.ndim!=2 or len(a)<4 or a.shape[1]<36: return 0.0
        body=a[:,:36].reshape(len(a),9,4)
        xy=body[:,:,:2]; vis=body[:,:,3]
        idx=[0,1,2,3,4,5,6]
        xy=xy[:,idx,:]; vis=vis[:,idx]
        valid=vis.mean(axis=0)>0.45
        if not np.any(valid): valid=np.ones(len(idx),dtype=bool)
        xy=xy[:,valid,:]
        delta=np.linalg.norm(np.diff(xy,axis=0),axis=2)
        return float(np.mean(np.sum(delta,axis=0)))

    def _baseline_strength(self,label):
        vals=[float(x) for x in self.strength_baselines.get(label,[]) if float(x)>1e-6]
        return float(np.median(vals)) if vals else 0.0

    def _drift_state(self,label,current_strength):
        base=self._baseline_strength(label)
        ratio=float(current_strength/base) if base>1e-6 else 1.0
        prior=self.drift_history.get(label,[])[-10:]
        prior_ratios=[float(x.get('ratio',1.0)) for x in prior]
        smoothed=float(np.median(prior_ratios[-5:]+[ratio])) if prior_ratios else ratio
        drift=float(np.clip(1.0-smoothed,0.0,0.75))
        # Keep the user's explicit 50% confidence floor. Adapt only the
        # temporal stability and class-separation requirements.
        stable_required=5 if drift>=0.22 else 6
        margin_required=float(max(0.09,0.14-0.055*drift/0.75))
        return {'baseline':base,'current':float(current_strength),'ratio':ratio,'smoothed_ratio':smoothed,
                'drift':drift,'stable_required':stable_required,'margin_required':margin_required}

    def _remember_drift(self,label,state,confidence):
        row={'time':time.strftime('%H:%M:%S'),'ratio':round(float(state['ratio']),3),
             'strength':round(float(state['current']),4),'confidence':round(float(confidence),3)}
        hist=self.drift_history.setdefault(label,[])
        hist.append(row); self.drift_history[label]=hist[-30:]
        self.last_drift={'label':label,**state,'confidence':float(confidence),'adapted':state['drift']>=0.12}
        self._save()

    def start_recording(self,label,phrase,emergency=False):
        label=label.strip().upper()
        if not label: raise ValueError('Gesture name required')
        self.phrases[label]=phrase.strip() or label.title()
        if emergency:self.emergency_labels.add(label)
        else:self.emergency_labels.discard(label)
        self.recording={'label':label,'frames':[],'target':self.window,'batch_total':8,'batch_done':0,'mode':'auto_gesture','pause_frames':0,'pause_target':12}; self._save()
        return {'label':label,'target_frames':self.window,'examples':8}

    def start_auto_rest(self, count=8):
        """Capture multiple REST examples continuously with one button press."""
        count=int(max(1,min(12,count)))
        self.phrases['REST']='Rest / no intentional gesture'
        self.recording={'label':'REST','frames':[],'target':self.window,'batch_total':count,'batch_done':0,'mode':'auto_rest'}
        self._save()
        return {'label':'REST','examples':count,'target_frames':count*self.window}

    def _quality_feedback(self, label, vec, frames):
        """Immediate, interpretable feedback for one newly recorded example.

        Scores are heuristic coaching signals, not medical measurements.
        """
        label=str(label).upper(); strength=self.movement_strength(frames)
        rest=self._rest_baseline_strength()
        notes=[]; status='good'; score=100; nearest=None; sep_score=None
        # Movement-vs-rest check
        if label not in self.rest_labels and rest>1e-6:
            threshold=max(rest*1.55,rest+0.0045)
            if strength<threshold:
                status='weak'; score-=35
                notes.append('Movement is close to REST. Make this gesture a little larger or more distinct.')
        elif label in self.rest_labels:
            vals=[float(x) for k in self.rest_labels for x in self.strength_baselines.get(k,[]) if float(x)>=0]
            if len(vals)>=3:
                med=float(np.median(vals))
                if strength>max(med*1.8,med+0.006):
                    status='noisy'; score-=30
                    notes.append('This REST sample contains more movement than your other REST examples. Relax and record it again.')
        # Within-class consistency against prior examples
        idx=[i for i,l in enumerate(self.labels[:-1]) if str(l).upper()==label]
        if idx:
            X=np.asarray([self.samples[i] for i in idx],dtype=np.float32)
            center=X.mean(0); scale=X.std(0)+0.08
            dist=float(np.mean(np.clip(np.abs((vec-center)/scale),0,5)))
            consistency=int(np.clip(100-18*dist,0,100))
            if consistency<55:
                status='inconsistent' if status=='good' else status; score-=20
                notes.append('This example differs from your earlier '+label+' recordings. Try to repeat the same motion path and speed.')
        else:
            consistency=100
        # Separation from other learned classes
        other_labels=sorted({str(l).upper() for l in self.labels[:-1] if str(l).upper()!=label})
        best=999.0
        for other in other_labels:
            oi=[i for i,l in enumerate(self.labels[:-1]) if str(l).upper()==other]
            if not oi: continue
            X=np.asarray([self.samples[i] for i in oi],dtype=np.float32)
            center=X.mean(0); scale=X.std(0)+0.10
            d=float(np.mean(np.clip(np.abs((vec-center)/scale),0,5)))
            if d<best: best=d; nearest=other
        if nearest is not None:
            sep_score=int(np.clip(22*best,0,100))
            if sep_score<45 and label not in self.rest_labels:
                status='too_similar'; score-=30
                notes.append(f'This looks similar to {nearest}. Use a more different body part, direction, or movement size.')
        score=int(np.clip(score,0,100))
        if not notes: notes.append('Good training example. Keep the next recordings similar but not perfectly identical.')
        return {'status':status,'score':score,'movement_strength':round(float(strength),5),
                'consistency':int(consistency),'nearest_other':nearest,'separation':sep_score,'notes':notes}

    def cancel_recording(self): self.recording=None

    def add_frame(self,feature):
        if feature is None:return None
        self.buffer.append(feature)
        if self.recording is not None:
            # Between auto-captured gesture examples, insert a short gap so the user
            # can return toward REST before the next sample begins.
            pause=int(self.recording.get('pause_frames',0))
            batch_total=int(self.recording.get('batch_total',1)); batch_done=int(self.recording.get('batch_done',0))
            if pause>0:
                self.recording['pause_frames']=pause-1
                total_progress=(batch_done + 0.0)/batch_total
                return {'recording':True,'progress':total_progress,'batch':batch_total>1,'batch_done':batch_done,'batch_total':batch_total,'pause':True,'pause_remaining':pause-1}
            self.recording['frames'].append(feature); n=len(self.recording['frames'])
            if n>=self.recording['target']:
                frames=self.recording['frames']; vec=self.summarize(frames); label=self.recording['label']
                self.samples.append(vec); self.seq_samples.append(np.asarray(frames,dtype=np.float32)); self.labels.append(label)
                strength=self.movement_strength(frames)
                self.strength_baselines.setdefault(label,[]).append(round(strength,5))
                self.strength_baselines[label]=self.strength_baselines[label][-20:]
                feedback=self._quality_feedback(label,vec,frames)
                batch_done += 1
                self.model=None
                if batch_done < batch_total:
                    self.recording['frames']=[]; self.recording['batch_done']=batch_done; self.recording['pause_frames']=int(self.recording.get('pause_target',12))
                    self._save()
                    return {'recording':True,'batch':True,'batch_done':batch_done,'batch_total':batch_total,
                            'progress':batch_done/batch_total,'label':label,'quality':feedback}
                self.recording=None; self._save()
                return {'sample_saved':True,'label':label,'count':self.sample_counts().get(label,0),'quality':feedback,
                        'batch':batch_total>1,'batch_done':batch_done,'batch_total':batch_total}
            total_progress=(batch_done + n/self.recording['target'])/batch_total
            return {'recording':True,'progress':total_progress,'batch':batch_total>1,'batch_done':batch_done,'batch_total':batch_total}
        return None

    def sample_counts(self): return dict(Counter(self.labels))

    def _separation_report(self, X, y, classes):
        report=[]
        for i,a in enumerate(classes):
            xa=X[y==a]
            if len(xa)<2: continue
            ca=xa.mean(axis=0); sa=xa.std(axis=0)+0.05
            for b in classes[i+1:]:
                xb=X[y==b]
                if len(xb)<2: continue
                cb=xb.mean(axis=0); sb=xb.std(axis=0)+0.05
                pooled=(sa+sb)/2
                z=np.abs(ca-cb)/pooled
                sep=float(np.mean(np.clip(z,0,5)))
                score=int(np.clip(20*sep,0,100))
                if score<60:
                    report.append({'a':a,'b':b,'score':score,'warning':'too similar' if score<40 else 'some overlap'})
        return sorted(report,key=lambda r:r['score'])[:8]

    def train(self):
        counts=self.sample_counts()
        if len(counts)<2: raise ValueError('Teach at least two classes first, including REST/NONE.')
        rest_count=sum(counts.get(k,0) for k in self.rest_labels)
        if rest_count < 5:
            raise ValueError(f'Record REST/NONE first. Exactly 8 REST examples are recommended ({rest_count} recorded).')
        if min(counts.values())<8: raise ValueError('Record 8 examples of every class, including REST/NONE.')
        X=np.asarray(self.samples); y=np.asarray(self.labels)
        model=RandomForestClassifier(n_estimators=550,max_depth=18,min_samples_leaf=1,class_weight='balanced_subsample',random_state=42,n_jobs=-1)
        n_splits=min(5,min(counts.values())); cv=StratifiedKFold(n_splits=n_splits,shuffle=True,random_state=42)
        pred=cross_val_predict(model,X,y,cv=cv); classes=sorted(counts)
        cm=confusion_matrix(y,pred,labels=classes).tolist(); acc=float(accuracy_score(y,pred))
        model.fit(X,y); self.model=model; joblib.dump(model,self.model_path)

        # GRU is deliberately gated. Tiny personalized datasets are usually more reliable
        # with RF + kinematics than with a neural temporal model.
        non_rest=[k for k in counts if k not in self.rest_labels]
        min_non_rest=min([counts[k] for k in non_rest], default=0)
        gru_ready=(len(non_rest)>=2 and min_non_rest>=20 and len(self.seq_samples)==len(self.labels))
        if gru_ready:
            try: gru_metrics=self.gru.fit(self.seq_samples,self.labels)
            except Exception as e: gru_metrics={'available':self.gru.available,'trained':False,'reason':str(e)}
        else:
            self.gru.reset()
            gru_metrics={'available':self.gru.available,'trained':False,'reason':f'Optional GRU stays off below 20 examples per non-rest gesture; the reliable RF + kinematic path is used at the 8-example target (current minimum: {min_non_rest}).'}

        separation=self._separation_report(X,y,classes)
        quality_components=[acc]
        quality_components.append(min(1.0, rest_count/8.0))
        quality_components.append(min(1.0, min_non_rest/8.0) if non_rest else 0.0)
        quality=int(round(100*np.mean(quality_components)))
        warnings=[]
        if rest_count<8: warnings.append(f'Add more REST/NONE examples ({rest_count}/8 recommended).')
        if min_non_rest<8: warnings.append(f'Add more gesture examples (minimum class has {min_non_rest}; 8 recommended).')
        if separation: warnings.append('Some learned gestures overlap; review the pair warnings below.')

        self.metrics={'accuracy':acc,'classes':classes,'confusion_matrix':cm,'samples':len(y),'gru':gru_metrics,
                      'fusion':'Adaptive RF + kinematics' + (' + causal GRU' if gru_metrics.get('trained') else ''),
                      'quality_score':quality,'rest_examples':rest_count,'min_gesture_examples':min_non_rest,
                      'separation_warnings':separation,'warnings':warnings}
        self.buffer.clear(); self.pred_history.clear(); self.armed=False; self._save(); return self.metrics

    def _compose(self,label,phrase,now):
        # Prefix gesture creates a short composition window instead of speaking immediately.
        if label in self.prefixes:
            self.pending_prefix=label; self.pending_until=now+4.5
            return {'pending':True,'pending_text':self.prefixes[label]+' …'}
        if self.pending_prefix and now<=self.pending_until:
            prefix=self.prefixes[self.pending_prefix]; target=label.replace('_',' ').title()
            if self.pending_prefix=='CALL': composed=f'{prefix} {target}.'
            else: composed=f'{prefix} {target.lower()}.'
            self.pending_prefix=None; self.pending_until=0
            return {'phrase':composed,'chained':True}
        self.pending_prefix=None; self.pending_until=0
        return {'phrase':phrase,'chained':False}

    def _rf_probs(self, seq):
        vec=self.summarize(seq).reshape(1,-1); p=self.model.predict_proba(vec)[0]
        return {str(c):float(p[i]) for i,c in enumerate(self.model.classes_)}

    def _kinematic_probs(self, seq):
        """Compare live kinematic summary to each class centroid using a soft distance score."""
        if not self.seq_samples or len(self.seq_samples)!=len(self.labels): return None
        seq=self._apply_dead_zones(seq)
        live=kinematic_summary(seq)
        out={}
        for label in sorted(set(map(str,self.labels))):
            rows=[kinematic_summary(s) for s,l in zip(self.seq_samples,self.labels) if str(l)==label]
            if not rows: continue
            center=np.mean(rows,axis=0); scale=np.std(rows,axis=0)+0.05
            z=(live-center)/scale; dist=float(np.mean(np.clip(z*z,0,25)))
            out[label]=float(np.exp(-0.12*dist))
        tot=sum(out.values()) or 1.0
        return {k:v/tot for k,v in out.items()}

    def _fusion_weights(self, gru_available):
        counts=self.sample_counts(); non_rest=[v for k,v in counts.items() if k not in self.rest_labels]
        n=min(non_rest) if non_rest else 0
        # Small-data regime: trust the simple personalized model.
        if not gru_available or n<20: return {'rf':0.68,'kin':0.32,'gru':0.0}
        if n<35: return {'rf':0.55,'kin':0.25,'gru':0.20}
        return {'rf':0.45,'kin':0.20,'gru':0.35}

    def _fuse_prob_maps(self,rf,kin=None,gru=None):
        classes=sorted(set(rf)|set(kin or {})|set(gru or {}))
        weights=self._fusion_weights(bool(gru))
        if not kin:
            weights={'rf':0.75,'kin':0.0,'gru':0.25 if gru else 0.0}
        den=sum(weights.values()) or 1.0
        fused={}
        for c in classes:
            fused[c]=(weights['rf']*rf.get(c,0)+weights['kin']*(kin or {}).get(c,0)+weights['gru']*(gru or {}).get(c,0))/den
        tot=sum(fused.values()) or 1.0
        return {k:v/tot for k,v in fused.items()},weights

    def _rest_baseline_strength(self):
        vals=[]
        for label in self.rest_labels:
            vals.extend(float(x) for x in self.strength_baselines.get(label,[]) if float(x)>=0)
        return float(np.median(vals)) if vals else 0.0

    def _has_rest_class(self):
        if self.model is None: return False
        classes={str(c).upper() for c in getattr(self.model,'classes_',[])}
        return bool(classes & self.rest_labels)

    def _intentional_motion(self, current_strength):
        """Hard movement gate based on this user's own REST recordings.

        Prevents a borderline classifier prediction from speaking while the user
        is motionless. The threshold is deliberately conservative and body-normalized.
        """
        rest=self._rest_baseline_strength()
        if rest <= 1e-6:
            return False, rest, 0.0
        threshold=max(rest*1.80, rest+0.006)
        return bool(current_strength >= threshold), rest, threshold

    def infer(self, mode="direct"):
        if self.model is None or len(self.buffer)<self.window:return None
        # Old models without REST are unsafe for speech. Force recalibration
        # rather than guessing while the user is idle.
        if not self._has_rest_class():
            return {'stable':False,'speak':False,'rest':False,'needs_rest_training':True,
                    'phrase':'Train REST/NONE before using live speech.'}
        seq=self._apply_dead_zones(list(self.buffer))
        rf=self._rf_probs(seq); kin=self._kinematic_probs(seq)
        try: gru=self.gru.predict_proba(seq)
        except Exception: gru=None
        fused,weights=self._fuse_prob_maps(rf,kin,gru)
        ranked=sorted(fused.items(),key=lambda kv:kv[1],reverse=True)
        label,conf=ranked[0]; second=ranked[1][1] if len(ranked)>1 else 0.0; margin=float(conf-second)
        self.last_fusion={'label':label,'fused':round(float(conf),4),'weights':weights,
                          'rf':round(float(rf.get(label,0)),4),'kinematic':round(float((kin or {}).get(label,0)),4),
                          'gru':round(float((gru or {}).get(label,0)),4) if gru else None,
                          'metrics':movement_metrics(seq)}
        self.pred_history.append((label,conf,margin)); votes=Counter(x[0] for x in self.pred_history); stable_label,stable_n=votes.most_common(1)[0]
        stable_conf=float(np.mean([c for l,c,m in self.pred_history if l==stable_label])); stable_margin=float(np.mean([m for l,c,m in self.pred_history if l==stable_label]))
        current_strength=self.movement_strength(list(self.buffer))
        intentional,rest_strength,motion_threshold=self._intentional_motion(current_strength)
        drift=self._drift_state(stable_label,current_strength)
        req_n=drift['stable_required']; req_margin=drift['margin_required']
        is_rest=stable_label in self.rest_labels
        result={'label':label,'confidence':conf,'margin':margin,'stable':False,'ambiguous':margin<req_margin,'rest':is_rest,'phrase':self.phrases.get(label,label.title()),
                'drift':drift,'adaptive':drift['drift']>=0.12,'fusion':self.last_fusion,
                'armed':self.armed,'intentional_motion':intentional,
                'movement_strength':round(float(current_strength),5),
                'rest_strength':round(float(rest_strength),5),
                'motion_threshold':round(float(motion_threshold),5)}
        now=time.time()
        if is_rest and stable_n>=max(3,req_n-1):
            # REST is now an explicit re-arm state. Nothing can speak until this
            # state has been observed after launch or after the previous phrase.
            self.armed=True
            result.update({'stable':True,'rest':True,'speak':False,'armed':True,
                           'phrase':'Rest detected — ready for a deliberate gesture.'})
            self.pred_history.clear(); self.buffer.clear()
            return result
        if (self.armed and intentional and len(self.pred_history)>=req_n and stable_n>=req_n
                and stable_conf>=0.50 and stable_margin>=req_margin and now>=self.cooldown_until):
            # Switch-access mode needs the learned movement as an input event, not an
            # automatically spoken phrase. This lets one reliable movement control
            # a much larger communication board.
            if mode == 'switch':
                result.update({'label':stable_label,'confidence':stable_conf,'margin':stable_margin,
                               'stable':True,'switch_event':True,'speak':False,
                               'phrase':self.phrases.get(stable_label,stable_label.title()),
                               'emergency':stable_label in self.emergency_labels})
            else:
                base_phrase=self.phrases.get(stable_label,stable_label.title()); composed=self._compose(stable_label,base_phrase,now)
                result.update({'label':stable_label,'confidence':stable_conf,'margin':stable_margin,'stable':True,'emergency':stable_label in self.emergency_labels,**composed})
                if not composed.get('pending'):
                    result['speak']=True
                    self.history.append({'time':time.strftime('%H:%M:%S'),'label':stable_label,'phrase':result['phrase'],'confidence':round(stable_conf,3),'chained':bool(result.get('chained')),'emergency':bool(result.get('emergency'))})
                    self.history=self.history[-50:]; self._save()
            self._remember_drift(stable_label,drift,stable_conf)
            result['drift']=drift; result['adaptive']=drift['drift']>=0.12
            # Require the user to return to REST before another event can fire.
            self.armed=False
            result['armed']=False
            self.cooldown_until=now+1.4; self.pred_history.clear(); self.buffer.clear()
        elif not self.armed and not is_rest:
            result['blocked_reason']='Return to REST to arm the next gesture.'
        elif self.armed and not intentional and not is_rest:
            result['blocked_reason']='No deliberate movement detected.'
        return result

    def reset_all(self):
        self.samples=[]; self.seq_samples=[]; self.labels=[]; self.phrases={}; self.emergency_labels=set(); self.model=None; self.metrics={}; self.history=[]
        self.gru=GRUGestureModel(model_dir=self.model_dir); self.last_fusion=None
        self.strength_baselines={}; self.drift_history={}; self.last_drift=None
        self.buffer.clear(); self.pred_history.clear(); self.recording=None; self.pending_prefix=None; self.armed=False
        self.gru.reset()
        for p in [self.samples_path,self.seq_path,self.meta_path,self.model_path]:
            if p.exists():p.unlink()


    def reset_drift(self):
        self.drift_history={}; self.last_drift=None; self._save()

    def profile(self):
        baselines={k:round(self._baseline_strength(k),4) for k in self.strength_baselines}
        drift_summary={}
        for k,rows in self.drift_history.items():
            if rows:
                ratios=[float(x.get('ratio',1.0)) for x in rows[-5:]]
                drift_summary[k]={'current_ratio':round(float(np.median(ratios)),3),'samples':len(rows),
                                  'recent':[{'ratio':float(x.get('ratio',1.0)),'time':x.get('time','')} for x in rows[-8:]]}
        return {'counts':self.sample_counts(),'phrases':self.phrases,'emergency_labels':sorted(self.emergency_labels),'trained':self.model is not None,'history':self.history[-12:][::-1],'metrics':self.metrics,'recording':bool(self.recording),'prefixes':self.prefixes,
                'strength_baselines':baselines,'drift_summary':drift_summary,'last_drift':self.last_drift,'fusion':self.last_fusion,'gru_available':self.gru.available,'gru_trained':self.gru.model is not None,'rest_labels':sorted(self.rest_labels),'reliability':{'confidence_floor':0.50,'prediction_history':self.pred_history.maxlen,'dead_zone_body':0.0035,'dead_zone_hand':0.0045,'rest_gate':True,'armed':self.armed,'requires_return_to_rest':True}}
