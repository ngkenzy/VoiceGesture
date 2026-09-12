from __future__ import annotations
import json, time
from collections import Counter, deque
from pathlib import Path

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict


class MovementReliabilityEngine:
    """Personalized accessibility-control discovery.

    Model A ranks recorded candidate movements against the user's own neutral
    posture. Tiny-Movement Auto Discovery goes one level lower: it finds the
    single body signal (for example head tilt or wrist shift) that most cleanly
    separates a user's chosen tiny movement from rest, learns a personalized
    threshold, and can emit a live switch event without the main gesture model.
    """

    CHANNEL_LABELS = {
        'head_tilt_deg': 'Head tilt',
        'shoulder_roll_deg': 'Shoulder tilt',
        'left_wrist_x': 'Left wrist side shift',
        'left_wrist_y': 'Left wrist vertical shift',
        'right_wrist_x': 'Right wrist side shift',
        'right_wrist_y': 'Right wrist vertical shift',
        'left_elbow_y': 'Left elbow vertical shift',
        'right_elbow_y': 'Right elbow vertical shift',
    }
    CHANNEL_FLOORS = {
        'head_tilt_deg': 1.5,
        'shoulder_roll_deg': 1.5,
        'left_wrist_x': .025, 'left_wrist_y': .025,
        'right_wrist_x': .025, 'right_wrist_y': .025,
        'left_elbow_y': .02, 'right_elbow_y': .02,
    }

    def __init__(self, data_dir='data', window=18):
        self.data_dir = Path(data_dir); self.data_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.data_dir / 'movement_reliability.npz'
        self.meta_path = self.data_dir / 'movement_reliability.json'
        self.window = window
        self.samples = []; self.labels = []; self.clip_metrics = []; self.tiny_summaries = []
        self.recording = None; self.results = {}; self.tiny_profile = {}
        self.live_values = deque(maxlen=5); self.tiny_active_frames = 0; self.tiny_release_frames = 0
        self.tiny_armed = True; self.tiny_cooldown_until = 0.0
        self._load()

    def _load(self):
        if self.path.exists():
            z = np.load(self.path, allow_pickle=True)
            self.samples = list(z['X']); self.labels = [str(x) for x in z['y']]
            if 'clip_metrics' in z: self.clip_metrics = list(z['clip_metrics'])
            if 'tiny_summaries' in z: self.tiny_summaries = list(z['tiny_summaries'])
        while len(self.clip_metrics) < len(self.samples): self.clip_metrics.append({})
        while len(self.tiny_summaries) < len(self.samples): self.tiny_summaries.append({})
        if self.meta_path.exists():
            try:
                meta = json.loads(self.meta_path.read_text())
                self.results = meta.get('results', {})
                self.tiny_profile = meta.get('tiny_profile', {})
            except Exception:
                self.results = {}; self.tiny_profile = {}

    def _save(self):
        if self.samples:
            np.savez_compressed(self.path, X=np.asarray(self.samples), y=np.asarray(self.labels, dtype=object),
                clip_metrics=np.asarray(self.clip_metrics, dtype=object), tiny_summaries=np.asarray(self.tiny_summaries, dtype=object))
        self.meta_path.write_text(json.dumps({'results': self.results, 'tiny_profile': self.tiny_profile}, indent=2))

    @staticmethod
    def summarize(seq):
        a=np.asarray(seq,dtype=np.float32); target=18
        old=np.linspace(0,1,len(a)); new=np.linspace(0,1,target)
        interp=np.vstack([np.interp(new,old,a[:,i]) for i in range(a.shape[1])]).T; d=np.diff(interp,axis=0)
        stats=np.concatenate([interp.mean(0),interp.std(0),interp.min(0),interp.max(0),interp[-1]-interp[0],np.abs(d).mean(0),np.abs(d).max(0)])
        return np.concatenate([stats,interp[[0,4,8,13,17]].reshape(-1)]).astype(np.float32)

    @staticmethod
    def movement_metrics(seq):
        a=np.asarray(seq,dtype=np.float32); body=a[:,:36].reshape(len(a),9,4); xy=body[:,:,:2]; vis=body[:,:,3]
        idx=[0,1,2,3,4,5,6]; xy=xy[:,idx,:]; vis=vis[:,idx]; valid=vis.mean(axis=0)>.45
        if not np.any(valid): valid=np.ones(len(idx),dtype=bool)
        xy=xy[:,valid,:]; span=np.linalg.norm(xy.max(axis=0)-xy.min(axis=0),axis=1); delta=np.linalg.norm(np.diff(xy,axis=0),axis=2)
        return {'amplitude':float(np.median(span)),'path':float(np.mean(np.sum(delta,axis=0))),
                'speed':float(np.mean(delta)),'smoothness':float(1/(1+np.std(delta)*10))}

    @classmethod
    def frame_channels(cls, feature):
        if feature is None or len(feature) < 36: return None
        body=np.asarray(feature[:36],dtype=np.float32).reshape(9,4)
        nose,ls,rs,le,re,lw,rw,lh,rh=body
        shoulder_vec=rs[:2]-ls[:2]
        shoulder_roll=float(np.degrees(np.arctan2(shoulder_vec[1], max(abs(shoulder_vec[0]),1e-5))))
        mid=(ls[:2]+rs[:2])/2; nv=nose[:2]-mid
        head_tilt=float(np.degrees(np.arctan2(nv[0], max(-nv[1],1e-5))))
        return {
            'head_tilt_deg': head_tilt,
            'shoulder_roll_deg': shoulder_roll,
            'left_wrist_x': float(lw[0]), 'left_wrist_y': float(lw[1]),
            'right_wrist_x': float(rw[0]), 'right_wrist_y': float(rw[1]),
            'left_elbow_y': float(le[1]), 'right_elbow_y': float(re[1]),
        }

    @classmethod
    def tiny_summary(cls, seq):
        rows=[cls.frame_channels(x) for x in seq]; rows=[r for r in rows if r]
        if not rows: return {}
        out={}
        for k in cls.CHANNEL_LABELS:
            v=np.asarray([r[k] for r in rows],dtype=float)
            out[k]={'mean':float(np.median(v)),'min':float(np.min(v)),'max':float(np.max(v)),
                    'range':float(np.max(v)-np.min(v)),'std':float(np.std(v))}
        return out

    def start_recording(self,label):
        label=(label or '').strip().upper()
        if not label: raise ValueError('Movement name required.')
        self.recording={'label':label,'frames':[],'target':self.window}
        return {'label':label,'target_frames':self.window}

    def cancel_recording(self): self.recording=None

    def add_frame(self,feature):
        if feature is None or self.recording is None: return None
        self.recording['frames'].append(feature); n=len(self.recording['frames'])
        if n < self.recording['target']: return {'recording':True,'progress':n/self.recording['target']}
        seq=self.recording['frames']; label=self.recording['label']
        self.samples.append(self.summarize(seq)); self.labels.append(label); self.clip_metrics.append(self.movement_metrics(seq)); self.tiny_summaries.append(self.tiny_summary(seq))
        self.recording=None; self.results={}
        if label in ('NEUTRAL','DISCOVER'): self.tiny_profile={}
        self._save(); return {'sample_saved':True,'label':label,'count':self.counts().get(label,0)}

    def counts(self): return dict(Counter(self.labels))

    @staticmethod
    def _repeatability(X):
        if len(X)<2:return 0.0
        X=np.asarray(X,dtype=np.float32); mu=X.mean(0); scale=X.std(0)+1e-3; z=(X-mu)/scale
        return float(np.clip(1-np.mean(np.sqrt(np.mean(z*z,axis=1)))/2.5,0,1))

    def analyze(self):
        counts=self.counts()
        if counts.get('NEUTRAL',0)<3: raise ValueError('Record at least 3 NEUTRAL/rest examples first.')
        candidates=[x for x in counts if x!='NEUTRAL' and counts[x]>=3]
        if not candidates: raise ValueError('Record at least 3 examples of one candidate movement.')
        Xall=np.asarray(self.samples,dtype=np.float32); yall=np.asarray(self.labels); outputs={}
        for label in candidates:
            mask=np.isin(yall,['NEUTRAL',label]); X=Xall[mask]; y=(yall[mask]==label).astype(int); min_class=int(min(np.sum(y==0),np.sum(y==1))); n_splits=min(4,min_class)
            pipe=LinearDiscriminantAnalysis(solver='lsqr',shrinkage='auto'); cv=StratifiedKFold(n_splits=n_splits,shuffle=True,random_state=42)
            pred=cross_val_predict(pipe,X,y,cv=cv); detect=float(balanced_accuracy_score(y,pred)); pipe.fit(X,y)
            idxs=[i for i,l in enumerate(self.labels) if l==label]; candX=[self.samples[i] for i in idxs]; metrics=[self.clip_metrics[i] for i in idxs]
            repeat=self._repeatability(candX); amplitude=float(np.median([m.get('amplitude',0) for m in metrics])); path=float(np.median([m.get('path',0) for m in metrics]))
            speeds=[m.get('speed',0) for m in metrics]; speed_cv=float(np.std(speeds)/(np.mean(speeds)+1e-5)); speed_stability=float(np.clip(1-speed_cv,0,1))
            effort=float(np.clip(path/3,0,1)); control_range=float(np.clip(amplitude/1.2,0,1))
            score=100*(.48*detect+.22*repeat+.12*speed_stability+.10*control_range+.08*(1-effort))
            outputs[label]={'score':round(float(score),1),'detection_reliability':round(detect,3),'repeatability':round(repeat,3),'speed_stability':round(speed_stability,3),'control_range':round(control_range,3),'effort':round(effort,3),'samples':counts[label]}
        ranked=sorted(outputs,key=lambda k:outputs[k]['score'],reverse=True); self.results={'recommended':ranked[0],'ranking':ranked,'movements':outputs,'neutral_samples':counts['NEUTRAL']}; self._save(); return self.results

    def analyze_tiny(self):
        counts=self.counts()
        if counts.get('NEUTRAL',0)<3: raise ValueError('Record at least 3 NEUTRAL examples first.')
        if counts.get('DISCOVER',0)<3: raise ValueError('Record at least 3 DISCOVER examples of the tiny movement you want to use.')
        neutral=[self.tiny_summaries[i] for i,l in enumerate(self.labels) if l=='NEUTRAL' and self.tiny_summaries[i]]
        discover=[self.tiny_summaries[i] for i,l in enumerate(self.labels) if l=='DISCOVER' and self.tiny_summaries[i]]
        if len(neutral)<3 or len(discover)<3: raise ValueError('Not enough usable pose samples for tiny-movement discovery.')
        ranked=[]
        for ch,label in self.CHANNEL_LABELS.items():
            nmeans=np.asarray([x[ch]['mean'] for x in neutral],float); nrange=np.asarray([x[ch]['range'] for x in neutral],float)
            base=float(np.median(nmeans)); noise=max(float(np.median(np.abs(nmeans-base)))*1.4826, float(np.median(nrange))/2, self.CHANNEL_FLOORS[ch])
            excursions=[]
            for d in discover:
                pos=d[ch]['max']-base; neg=d[ch]['min']-base; excursions.append(pos if abs(pos)>=abs(neg) else neg)
            exc=np.asarray(excursions,float); direction=1 if np.median(exc)>=0 else -1; signed=direction*exc
            consistency=float(np.mean(signed>0)); med=float(np.median(np.maximum(signed,0))); mad=float(np.median(np.abs(signed-np.median(signed))))
            repeat=float(np.clip(1-mad/(med+noise+1e-6),0,1)); separation=float(np.clip(med/(noise*5),0,1));
            # Favor small but clean controls: once separation is adequate, smaller excursion gets a slight bonus.
            tiny_bonus=float(np.clip(1-med/(12 if 'deg' in ch else 1.2),0,1))
            score=100*(.52*separation+.28*repeat+.14*consistency+.06*tiny_bonus)
            threshold=max(noise*2.2, med*.45)
            dirword=self._direction_word(ch,direction)
            ranked.append({'channel':ch,'label':label,'direction':direction,'direction_word':dirword,'score':round(score,1),
                           'baseline':base,'noise':noise,'median_excursion':med,'threshold':threshold,'repeatability':round(repeat,3),
                           'consistency':round(consistency,3),'separation':round(separation,3),'unit':'°' if 'deg' in ch else 'body units'})
        ranked.sort(key=lambda x:x['score'],reverse=True); best=ranked[0]
        if best['score']<42: raise ValueError('I could not find a reliable tiny movement yet. Try making the same small movement more consistently.')
        self.tiny_profile={'enabled':True,'channel':best['channel'],'label':best['label'],'direction':best['direction'],
                           'direction_word':best['direction_word'],'baseline':best['baseline'],'noise':best['noise'],
                           'median_excursion':best['median_excursion'],'threshold':best['threshold'],'score':best['score'],
                           'unit':best['unit'],'ranking':ranked[:5],'samples':counts.get('DISCOVER',0)}
        self.live_values.clear(); self.tiny_active_frames=0; self.tiny_release_frames=0; self.tiny_armed=True; self._save(); return self.tiny_profile

    @staticmethod
    def _direction_word(ch,direction):
        if ch=='head_tilt_deg': return 'right' if direction>0 else 'left'
        if ch=='shoulder_roll_deg': return 'clockwise' if direction>0 else 'counter-clockwise'
        if ch.endswith('_x'): return 'right' if direction>0 else 'left'
        if ch.endswith('_y'): return 'down' if direction>0 else 'up'
        return 'positive' if direction>0 else 'negative'

    def tiny_live(self,feature):
        p=self.tiny_profile
        if not p.get('enabled') or feature is None or self.recording is not None: return None
        row=self.frame_channels(feature)
        if not row:return None
        val=float(row[p['channel']]); self.live_values.append(val)
        smooth=float(np.median(self.live_values)); signed=float(p['direction']*(smooth-p['baseline'])); threshold=float(p['threshold'])
        strength=float(max(0,signed)/(max(p['median_excursion'],1e-6)))
        active=signed>=threshold
        now=time.time(); event=False
        if active:
            self.tiny_active_frames+=1; self.tiny_release_frames=0
            if self.tiny_armed and self.tiny_active_frames>=3 and now>=self.tiny_cooldown_until:
                event=True; self.tiny_armed=False; self.tiny_cooldown_until=now+1.0
        else:
            self.tiny_active_frames=0
            if signed < threshold*.35:
                self.tiny_release_frames+=1
                if self.tiny_release_frames>=3:self.tiny_armed=True
            else:self.tiny_release_frames=0
        return {'enabled':True,'event':event,'active':active,'armed':self.tiny_armed,'channel':p['channel'],'label':p['label'],
                'direction_word':p['direction_word'],'strength':round(strength,3),'value':round(smooth,4),
                'threshold_progress':round(float(np.clip(signed/max(threshold,1e-6),0,1.5)),3),'score':p['score']}

    def reset_tiny(self):
        self.tiny_profile={}; self.live_values.clear(); self.tiny_active_frames=0; self.tiny_release_frames=0; self.tiny_armed=True; self._save()

    def reset(self):
        self.samples=[]; self.labels=[]; self.clip_metrics=[]; self.tiny_summaries=[]; self.results={}; self.tiny_profile={}; self.recording=None
        self.live_values.clear(); self.tiny_armed=True
        for p in (self.path,self.meta_path):
            if p.exists():p.unlink()

    def profile(self):
        return {'counts':self.counts(),'results':self.results,'tiny_profile':self.tiny_profile,'recording':bool(self.recording)}
