const video=document.getElementById('video'),overlay=document.getElementById('overlay'),canvas=document.getElementById('capture');
const ctx=canvas.getContext('2d');let sending=false,speechOn=true,timer=null,profile={},mode='direct',scanTimer=null,scanIndex=0,currentBoard='HOME',highlightStarted=Date.now(),phraseJourneyStart=null,phraseSelections=0,handsUpSince=null,handsUpLatched=false,handsDownSince=Date.now();const $=id=>document.getElementById(id);
const efficiency=JSON.parse(localStorage.getItem('gvEfficiency')||'{"phrases":0,"totalMs":0,"totalSelections":0,"scanMs":2000,"timings":[]}');
const BOARDS={
 HOME:[['GREETINGS','👋','Greetings'],['NEEDS','💧','Needs'],['COMFORT','🛏️','Comfort'],['PEOPLE','👨‍👩‍👧','People'],['RESPONSES','💬','Responses'],['HELP_NOW','⚠️','I need help now']],
 GREETINGS:[['SAY','👋','Hello!'],['SAY','🌅','Good morning.'],['SAY','🌤️','Good afternoon.'],['SAY','🌙','Good evening.'],['SAY','🙂','How are you?'],['SAY','😊','It is nice to see you.'],['SAY','🙏','Thank you.'],['SAY','👋','Goodbye.']],
 NEEDS:[['SAY','💧','Can I have some water, please?'],['SAY','🍴','I am hungry.'],['SAY','🚻','I need to use the bathroom.'],['SAY','💊','I need my medication.'],['SAY','😴','I need to rest.'],['SAY','🧃','I would like another drink.'],['SAY','🧻','I need help getting cleaned up.'],['SAY','👕','I need help getting dressed.'],['SAY','🧼','I would like to wash up.']],
 COMFORT:[['SAY','🛏️','Please help me reposition.'],['SAY','🥶','I am too cold.'],['SAY','🥵','I am too hot.'],['SAY','😣','I am uncomfortable.'],['SAY','🔇','Please make it quieter.'],['SAY','💡','Please adjust the lights.'],['SAY','🪑','Please help me sit up.'],['SAY','🛌','Please help me lie down.'],['SAY','⏸️','I need a break.']],
 PEOPLE:[['SAY','👨‍👩‍👧','Please call my family.'],['SAY','🧑‍⚕️','Please get my nurse or caregiver.'],['SAY','🧑','I would like to speak with someone.'],['SAY','🏠','I am ready to go home.'],['SAY','📞','Please help me make a phone call.'],['SAY','👋','I would like some company.'],['SAY','🤫','I would like some privacy.'],['SAY','📝','Please tell my caregiver I need something.']],
 RESPONSES:[['SAY','✅','Yes.'],['SAY','❌','No.'],['SAY','🔁','Please say that again.'],['SAY','⏳','Please give me a moment.'],['SAY','🙏','Thank you.'],['SAY','👍','That is correct.'],['SAY','👎','That is not what I meant.'],['SAY','❓','I have a question.'],['SAY','🛑','Please stop.']]
};
async function api(url,body){const r=await fetch(url,{method:body?'POST':'GET',headers:{'Content-Type':'application/json'},body:body?JSON.stringify(body):undefined});const j=await r.json();if(!r.ok||j.ok===false)throw new Error(j.error||'Request failed');return j}
async function startCamera(){try{const s=await navigator.mediaDevices.getUserMedia({video:{width:{ideal:960},height:{ideal:600}},audio:false});video.srcObject=s;await video.play();$('cameraBadge').textContent='Camera live';timer=setInterval(sendFrame,125)}catch(e){$('cameraBadge').textContent='Camera permission needed';$('cameraBadge').title=e.message}}
async function sendFrame(){
  if(sending||video.readyState<2)return;
  sending=true;
  try{
    canvas.width=640;canvas.height=400;
    ctx.save();ctx.translate(canvas.width,0);ctx.scale(-1,1);ctx.drawImage(video,0,0,canvas.width,canvas.height);ctx.restore();
    const j=await api('/api/frame',{image:canvas.toDataURL('image/jpeg',.72),mode});
    if(j.image)overlay.src=j.image;
    renderPrediction(j.prediction);renderTinyPrediction(j.tiny_prediction);handleHandsUpSwitch(j.status);
    if(j.event){
      if(j.event.quality)renderCoach(j.event.quality);
      if(j.event.recording){
        $('progressFill').style.width=Math.round(j.event.progress*100)+'%';
        if(j.event.batch){
          const lbl=($('label').value||'GESTURE').toUpperCase();
          if(j.event.pause){
            $('recordStatus').textContent=`${lbl}: ${j.event.batch_done||0}/${j.event.batch_total} saved · return to REST…`;
          }else{
            $('recordStatus').textContent=`${lbl}: auto-recording ${Math.min((j.event.batch_done||0)+1,j.event.batch_total)}/${j.event.batch_total} · ${Math.round(j.event.progress*100)}%`;
          }
          $('judgeStatus').textContent=$('recordStatus').textContent;
        }else{
          $('recordStatus').textContent=`Recording… ${Math.round(j.event.progress*100)}%`;
        }
      }
      if(j.event.sample_saved){
        $('progressFill').style.width='100%';
        $('recordStatus').textContent=j.event.batch?`Auto REST complete: ${j.event.batch_done}/${j.event.batch_total} saved.`:`Saved example ${j.event.count} for ${j.event.label}.`;
        if(j.event.batch)$('judgeStatus').textContent='REST calibration complete. Now teach SELECT eight times.';
        if(j.profile){profile=j.profile;renderProfile()}else loadProfile();
      }
    }
    if(j.movement_event){
      if(j.movement_event.recording){
        $('movementProgress').style.width=Math.round(j.movement_event.progress*100)+'%';
        $('movementStatus').textContent=`Recording movement… ${Math.round(j.movement_event.progress*100)}%`;
      }
      if(j.movement_event.sample_saved){
        $('movementProgress').style.width='100%';
        $('movementStatus').textContent=`Saved movement sample ${j.movement_event.count} for ${j.movement_event.label}.`;
        if(j.profile){profile=j.profile;renderProfile()}
      }
    }
  }catch(e){console.error(e)}finally{sending=false}
}
function handleHandsUpSwitch(status){
 if(mode!=='switch'||!$('useHandsUpControl')||!$('useHandsUpControl').checked)return;
 const up=!!(status&&status.hands_up),now=Date.now();
 if(up){
   handsDownSince=null;
   if(handsUpSince===null)handsUpSince=now;
   const held=now-handsUpSince;
   const src=(status&&status.switch_source&&status.switch_source!=='none')?status.switch_source:'vision';
   $('boardStatus').textContent=handsUpLatched?`Hand detected via ${src} · lower hand to re-arm`:`Hand detected via ${src} · hold ${Math.max(0,250-held)} ms`;
   if(!handsUpLatched && held>=250){
     handsUpLatched=true;
     $('spokenPhrase').textContent='Hand raised → SELECT';
     $('boardStatus').textContent=`SELECT fired · ${src} · lower hand to re-arm`;
     selectHighlighted();
   }
 }else{
   handsUpSince=null;
   if(handsDownSince===null)handsDownSince=now;
   if(handsUpLatched && now-handsDownSince>=300){
     handsUpLatched=false;
     $('boardStatus').textContent='Switch re-armed. Raise a hand to select.';
   }
 }
}

function renderFusion(f){if(!f)return;$('fusionState').textContent=(f.label||'Movement')+' detected';$('fusionRF').textContent=f.rf!=null?`${Math.round(f.rf*100)}%`:'—';$('fusionKin').textContent=f.kinematic!=null?`${Math.round(f.kinematic*100)}%`:'—';$('fusionGRU').textContent=f.gru!=null?`${Math.round(f.gru*100)}%`:'OFF';$('fusionFinal').textContent=f.fused!=null?`${Math.round(f.fused*100)}%`:'—';const m=f.metrics||{};$('kinHead').textContent=m.head_range_deg!=null?`${m.head_range_deg.toFixed(1)}°`:'—';$('kinSpeed').textContent=m.peak_wrist_speed!=null?m.peak_wrist_speed.toFixed(3):'—';$('kinSmooth').textContent=m.smoothness!=null?m.smoothness.toFixed(2):'—'}

function renderCoach(q){if(!q)return;const score=q.score??0;$('coachScore').textContent=`${score}/100 · ${String(q.status||'').replace('_',' ')}`;$('coachBox').classList.toggle('warn',score<65);$('coachBox').classList.toggle('good',score>=80);const bits=[];if(q.consistency!=null)bits.push(`consistency ${q.consistency}/100`);if(q.separation!=null&&q.nearest_other)bits.push(`vs ${q.nearest_other}: ${q.separation}/100`);$('coachMessage').textContent=(q.notes||[]).join(' ') + (bits.length?' · '+bits.join(' · '):'');}
function updateJudgeProgress(){const c=profile.counts||{};const rest=c.REST||c.NONE||c.NEUTRAL||0;const select=c.SELECT||0;let n=0,h='1. Auto-calibrate REST';if(rest>=8){n=1;h='2. Teach SELECT eight times'}if(rest>=8&&select>=8){n=2;h='3. Train and enter Switch Board'}if(rest>=8&&select>=8&&profile.trained){n=3;h='Ready for the live demo'}$('judgeProgress').textContent=`${n} / 3`;$('judgeHint').textContent=h;}
function renderPrediction(p){if(!p)return;if(p.needs_rest_training){$('gestureName').textContent='REST REQUIRED';$('confidence').textContent='Speech locked for safety';$('meterFill').style.width='0%';$('spokenPhrase').textContent='Train REST / NONE before live speech can trigger.';$('chainHint').textContent='The false-trigger guard requires a neutral baseline.';return}$('gestureName').textContent=p.rest?'REST / NONE':p.label;$('confidence').textContent=`Confidence ${Math.round((p.confidence||0)*100)}% · separation ${Math.round((p.margin||0)*100)}%`;$('meterFill').style.width=Math.round((p.confidence||0)*100)+'%';renderLiveDrift(p);renderFusion(p.fusion);if(p.rest){$('spokenPhrase').textContent='Rest detected — armed and ready.';$('chainHint').textContent='Speech stays locked until a deliberate movement leaves the learned REST baseline.';return}if(p.blocked_reason&&!p.stable){$('spokenPhrase').textContent=p.blocked_reason;$('chainHint').textContent=p.armed?'Waiting for deliberate movement above the REST threshold.':'Return to REST before the next phrase can fire.';}else if(p.ambiguous&&!p.stable)$('spokenPhrase').textContent='Not sure yet — please repeat the movement.';
 if(mode==='switch'){if(($('useHandsUpControl')&&$('useHandsUpControl').checked)||$('useTinyControl').checked)return;if(p.stable&&p.switch_event){const expected=$('selectGesture').value;if(p.label===expected){$('spokenPhrase').textContent=`${p.label} → SELECT`;selectHighlighted()}else{$('spokenPhrase').textContent=`Saw ${p.label}. Waiting for ${expected}.`}}return}
 if(p.pending){$('spokenPhrase').textContent=p.pending_text;$('chainHint').textContent='Prefix stored. Perform the second gesture within 4.5 seconds.';return}
 if(p.stable&&p.phrase){$('spokenPhrase').textContent='“'+p.phrase+'”';$('chainHint').textContent=p.chained?'Gesture chain recognized.':'The system waited for a stable, separated prediction before speaking.';if(p.emergency)flashEmergency();if(p.speak&&speechOn){speak(p.phrase,p.emergency);setTimeout(loadProfile,400)}}}
function flashEmergency(){$('heroCard').classList.add('emergency');$('emergencyBanner').classList.add('show');setTimeout(()=>{$('heroCard').classList.remove('emergency');$('emergencyBanner').classList.remove('show')},3500)}
let elevenAudio=null;
function setVoiceStatus(text,ok=true){
 const el=$('voiceProviderStatus');
 if(!el)return;
 el.textContent=text;
 el.className='voice-provider-status '+(ok?'ok':'warn');
}
async function speak(text,urgent=false){
 if(!speechOn)return;
 speechSynthesis.cancel();
 try{
   if(elevenAudio){elevenAudio.pause();elevenAudio=null}
   setVoiceStatus('ElevenLabs: generating…',true);
   const r=await fetch('/api/tts',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text,urgent})});
   if(!r.ok){
     let msg='ElevenLabs unavailable';
     try{const j=await r.json(); if(j&&j.error)msg=j.error}catch(_){}
     throw new Error(msg);
   }
   const blob=await r.blob();
   if(!blob.size)throw new Error('ElevenLabs returned empty audio');
   const url=URL.createObjectURL(blob);
   elevenAudio=new Audio(url);
   elevenAudio.onended=()=>{URL.revokeObjectURL(url);elevenAudio=null};
   elevenAudio.onerror=()=>{URL.revokeObjectURL(url);setVoiceStatus('Browser fallback: ElevenLabs audio playback failed',false);browserSpeak(text,urgent,false)};
   await elevenAudio.play();
   setVoiceStatus('✓ ElevenLabs voice played',true);
   return;
 }catch(e){
   console.warn('ElevenLabs TTS failed:',e);
   setVoiceStatus('Browser fallback: '+(e?.message||'ElevenLabs failed'),false);
   browserSpeak(text,urgent,false);
 }
}
function browserSpeak(text,urgent=false,update=true){
 if(!speechOn)return;
 if(update)setVoiceStatus('Browser voice',false);
 speechSynthesis.cancel();
 const u=new SpeechSynthesisUtterance(text);u.rate=urgent?.85:.95;u.volume=1;speechSynthesis.speak(u)
}
async function checkElevenLabs(){
 try{
   const r=await fetch('/api/tts/status');
   const j=await r.json();
   if(j.ok){
     setVoiceStatus('ElevenLabs key loaded · TTS not tested yet',true);
   }else setVoiceStatus('ElevenLabs not configured: '+(j.error||'unknown error'),false);
 }catch(e){setVoiceStatus('ElevenLabs status check failed',false)}
}
async function testElevenLabs(){
 setVoiceStatus('Testing ElevenLabs TTS…',true);
 try{
   const r=await fetch('/api/tts',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text:'GestureVoice ElevenLabs test. Your voice connection is working.',urgent:false})});
   if(!r.ok){
     let msg='ElevenLabs TTS test failed';
     try{const j=await r.json(); msg=(j.error||msg)+(j.hint?' — '+j.hint:'')}catch(_){ }
     throw new Error(msg);
   }
   const blob=await r.blob();
   const url=URL.createObjectURL(blob);
   const a=new Audio(url);
   a.onended=()=>URL.revokeObjectURL(url);
   await a.play();
   setVoiceStatus('✓ ElevenLabs TTS verified',true);
 }catch(e){
   setVoiceStatus('ElevenLabs test failed: '+(e?.message||'unknown error'),false);
 }
}
async function loadProfile(){profile=await api('/api/profile');renderProfile();checkElevenLabs()}
function renderReliability(m){m=m||{};const score=m.quality_score;$('reliabilityScore').textContent=score!=null?`${score}/100`:'—';$('reliabilityHint').textContent=score!=null?(score>=85?'Demo-ready training quality.':score>=65?'Usable, but more examples should improve reliability.':'Needs more calibration data.'):'Train the personalized model to score reliability.';$('restExamples').textContent=`${m.rest_examples||0} / 8`;$('minGestureExamples').textContent=`${m.min_gesture_examples||0} / 8`;$('stabilityWindow').textContent='6 votes';const gm=m.gru||{};$('gruPolicy').textContent=gm.trained?'Enabled':'Gated';const warnings=[...(m.warnings||[])];for(const x of (m.separation_warnings||[])){warnings.push(`${x.a} ↔ ${x.b}: separation ${x.score}/100 (${x.warning})`)}$('reliabilityWarnings').innerHTML=warnings.length?warnings.map((w,i)=>`<div class="movement-result ${i===0?'best':''}"><strong>${escapeHtml(w)}</strong></div>`).join(''):'<div class="movement-result best"><strong>✓ No major training warnings detected.</strong></div>'}
function renderProfile(){const c=profile.counts||{},ph=profile.phrases||{},em=new Set(profile.emergency_labels||[]),entries=Object.keys(c);$('sampleList').innerHTML=entries.length?entries.sort().map(k=>`<div class="sample-row"><span>${escapeHtml(k)}${em.has(k)?'<b class="emtag">EMERGENCY</b>':''}<small>${escapeHtml(ph[k]||'')}</small></span><span>${c[k]} / 8</span></div>`).join(''):'<div class="empty">Teach your first gesture to begin.</div>';$('modelBadge').textContent=profile.trained?'● Personalized model live':'Not trained';$('modelBadge').className='model-badge '+(profile.trained?'live':'');const m=profile.metrics||{};renderReliability(m);$('trainStatus').textContent=m.accuracy!=null?`Cross-validated accuracy: ${Math.round(m.accuracy*100)}% · ${m.samples} samples`:'For reliability: 8 REST/NONE examples + 8 examples per gesture.';$('qualityBox').innerHTML=m.accuracy!=null?`<strong>${Math.round(m.accuracy*100)}%</strong><span>validation accuracy</span><small>${(m.classes||[]).length} learned movements · 50% speech threshold</small>`:'';$('yoloxPill').textContent=profile.yolo11_enabled?(profile.yolo_model||'YOLO11s-pose'):'MediaPipe fallback';$('poseBackend').textContent=(profile.pose_backend||'MediaPipe Pose')+(profile.elevenlabs_enabled?' · ElevenLabs voice ready':' · Browser voice fallback');$('posePipeline').textContent=(profile.pose_backend||'MediaPipe Pose')+' + hands';const gm=(profile.metrics||{}).gru||{};$('gruStatus').textContent=profile.gru_trained?'causal GRU trained on temporal sequences':(profile.gru_available?'available — retrain model to enable':'install neural extras to enable');$('fusionBackend').textContent=(profile.metrics||{}).fusion||'RF + kinematics';if(profile.fusion)renderFusion(profile.fusion);const h=profile.history||[];$('historyList').innerHTML=h.length?h.map(x=>`<div class="history-item ${x.emergency?'urgent':''}"><strong>${x.chained?'🔗 ':''}${x.emergency?'⚠️ ':''}${escapeHtml(x.phrase)}</strong><span>${x.time} · ${Math.round(x.confidence*100)}%</span></div>`).join(''):'<div class="empty">No direct-mode phrases yet.</div>';renderMovementLab(profile.movement_lab||{});renderDriftProfile(profile);refreshGestureSelect(entries);updateJudgeProgress()}
function refreshGestureSelect(labels){const sel=$('selectGesture'),prior=sel.value||localStorage.getItem('gvSelectGesture')||'SELECT';sel.innerHTML=labels.length?labels.map(x=>`<option value="${escapeHtml(x)}">${escapeHtml(x)}</option>`).join(''):'<option value="SELECT">SELECT</option>';if([...sel.options].some(o=>o.value===prior))sel.value=prior;sel.onchange=()=>localStorage.setItem('gvSelectGesture',sel.value)}
function escapeHtml(s){return String(s).replace(/[&<>'"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[c]))}

function renderMovementLab(lab){
 renderTinyProfile(lab.tiny_profile||{});
 const counts=lab.counts||{},names=Object.keys(counts).sort();
 $('movementCounts').innerHTML=names.length?names.map(k=>`<div class="sample-row"><span>${escapeHtml(k)}<small>${k==='NEUTRAL'?'rest baseline':'candidate control'}</small></span><span>${counts[k]} / 3+</span></div>`).join(''):'<div class="empty">No movement calibration samples yet.</div>';
 const r=lab.results||{},moves=r.movements||{},ranking=r.ranking||[];
 if(r.recommended){$('movementRecommendation').classList.add('good');$('movementRecommendation').innerHTML=`<small>BEST CONTROL</small><strong>${escapeHtml(r.recommended)}</strong><span>${moves[r.recommended].score}/100 reliability score</span>`}else{$('movementRecommendation').classList.remove('good');$('movementRecommendation').innerHTML='<small>BEST CONTROL</small><strong>Not analyzed yet</strong><span>Record 3+ NEUTRAL examples and 3+ examples of each candidate.</span>'}
 $('movementResults').innerHTML=ranking.length?ranking.map((k,i)=>{const m=moves[k];return `<div class="move-score ${i===0?'best':''}"><div class="top"><strong>${i===0?'★ ':''}${escapeHtml(k)}</strong><span class="score">${m.score}</span></div><div class="metrics"><span><b>${Math.round(m.detection_reliability*100)}%</b>detection</span><span><b>${Math.round(m.repeatability*100)}%</b>repeatability</span><span><b>${Math.round(m.speed_stability*100)}%</b>speed stability</span><span><b>${Math.round((1-m.effort)*100)}%</b>low effort</span></div></div>`}).join(''):'';
 if(r.recommended){const sel=$('selectGesture');if([...sel.options].some(o=>o.value===r.recommended)){sel.value=r.recommended;localStorage.setItem('gvSelectGesture',r.recommended)}}
}

function renderTinyProfile(t){
 const box=$('tinyRecommendation'),btn=$('tinyEnableBtn');
 if(t&&t.enabled){
  box.classList.add('good');
  const amount=(t.median_excursion||0).toFixed(t.unit==='°'?1:3);
  box.innerHTML=`<small>AUTO-DISCOVERED CONTROL</small><strong>${escapeHtml(t.label)} ${escapeHtml(t.direction_word||'')}</strong><span>${amount}${t.unit||''} typical excursion · ${Math.round(t.score||0)}/100 signal score</span>`;
  btn.disabled=false;$('tinyLiveName').textContent=`${t.label} ${t.direction_word||''}`;$('tinyLiveDetail').textContent='Ready. Enable this signal in Switch Board and repeat the tiny movement to SELECT.';
  const rows=t.ranking||[];$('tinyRanking').innerHTML=rows.map((r,i)=>`<div class="tiny-rank ${i===0?'best':''}"><div class="top"><strong>${i===0?'★ ':''}${escapeHtml(r.label)} ${escapeHtml(r.direction_word||'')}</strong><span>${Math.round(r.score||0)}/100</span></div><small>${(r.median_excursion||0).toFixed(r.unit==='°'?1:3)}${r.unit||''} · repeatability ${Math.round((r.repeatability||0)*100)}% · consistency ${Math.round((r.consistency||0)*100)}%</small></div>`).join('');
 }else{
  box.classList.remove('good');box.innerHTML='<small>AUTO-DISCOVERED CONTROL</small><strong>Not discovered yet</strong><span>3× NEUTRAL + 3× DISCOVER</span>';btn.disabled=true;$('tinyRanking').innerHTML='';
 }
}
function renderTinyPrediction(t){
 if(!t||!t.enabled)return;
 const pct=Math.round(Math.min(1.5,t.threshold_progress||0)*100);$('tinyLiveBar').style.width=Math.min(100,pct)+'%';$('tinyLiveBar').parentElement.classList.toggle('trigger',!!t.active);
 $('tinyLiveName').textContent=`${t.label} ${t.direction_word||''} · ${Math.round((t.strength||0)*100)}% of learned movement`;
 $('tinyLiveDetail').textContent=t.active?'Tiny control detected. Release back toward neutral to re-arm.':'Move toward the learned direction to reach the trigger.';
 if(mode==='switch'&&$('useTinyControl').checked&&t.event){$('spokenPhrase').textContent='Tiny movement → SELECT';$('boardStatus').textContent='Tiny control selected highlighted choice.';selectHighlighted()}
}

function renderLiveDrift(p){
 if(!p||!p.drift)return;const d=p.drift,ratio=Math.max(0,Math.min(1.2,d.ratio||1)),pct=Math.round(ratio*100);
 $('strengthPercent').textContent=pct+'%';$('currentBar').style.width=Math.min(100,pct)+'%';$('driftGesture').textContent=(p.label||'Gesture')+' movement';
 const base=(d.baseline||0).toFixed(3),cur=(d.current||0).toFixed(3);$('driftDetail').textContent=`Baseline ${base} · current ${cur} · 50% confidence floor preserved`;
 const state=$('driftState');if(p.adaptive){state.classList.add('adapting');state.innerHTML='<small>ADAPTATION STATUS</small><strong>Adaptive mode active</strong><span>Movement is smaller than baseline; temporal requirements are adjusted within safe bounds.</span>'}else{state.classList.remove('adapting');state.innerHTML='<small>ADAPTATION STATUS</small><strong>Baseline mode</strong><span>Movement remains close to the user’s learned gesture strength.</span>'}
}
function renderDriftProfile(p){
 const last=p.last_drift;if(last){const ratio=Math.max(0,Math.min(1.2,last.smoothed_ratio||last.ratio||1)),pct=Math.round(ratio*100);$('strengthPercent').textContent=pct+'%';$('currentBar').style.width=Math.min(100,pct)+'%';$('driftGesture').textContent=escapeHtml(last.label)+' movement';$('driftDetail').textContent=`Recent movement retained ${pct}% of its learned baseline.`}
 const ds=p.drift_summary||{},key=last&&last.label?last.label:Object.keys(ds)[0];let rows=[];
 if(key&&ds[key]&&ds[key].recent){rows=ds[key].recent.map(x=>{const pct=Math.round(Math.max(0,Math.min(1.2,x.ratio||1))*100);return {pct,adapt:pct<88,time:x.time||''}})}
 $('driftHistory').innerHTML=rows.length?rows.map(x=>`<div class="drift-dot ${x.adapt?'adapt':''}" title="${x.time} · ${x.pct}% retained" style="height:${Math.max(10,Math.min(48,x.pct/2))}px"><span>${x.pct}%</span></div>`).join(''):'<div class="empty">No adaptive observations yet.</div>';
}
function totalBoardPhrases(){return Object.keys(BOARDS).filter(k=>k!=='HOME').reduce((n,k)=>n+BOARDS[k].filter(x=>x[0]==='SAY').length,0)+BOARDS.HOME.filter(x=>x[0]==='HELP_NOW').length}
function saveEfficiency(){localStorage.setItem('gvEfficiency',JSON.stringify(efficiency));renderEfficiency()}
function renderEfficiency(){const count=totalBoardPhrases();$('phraseCount').textContent=`${count} phrases`;$('successfulPhrases').textContent=efficiency.phrases||0;$('avgPhraseTime').textContent=efficiency.phrases?`${(efficiency.totalMs/efficiency.phrases/1000).toFixed(1)} sec`:'—';$('selectionsPerPhrase').textContent=efficiency.phrases?(efficiency.totalSelections/efficiency.phrases).toFixed(1):'—';$('adaptiveScanState').textContent=`${efficiency.scanMs||2000} ms`;$('adaptiveScanHint').textContent=$('adaptiveScan').checked?'Adjusts after each intentional selection.':'Automatic timing disabled.';$('phraseTimeHint').textContent=efficiency.phrases?'Measured from first category selection to spoken phrase.':'Complete a phrase in Switch Board to measure.';if($('scanSpeed'))$('scanSpeed').value=String(nearestSpeed(efficiency.scanMs||2000))}
function nearestSpeed(ms){const vals=[1800,1300,900];return vals.reduce((a,b)=>Math.abs(b-ms)<Math.abs(a-ms)?b:a,vals[0])}
function noteHighlight(){highlightStarted=Date.now()}
function adaptScanFromSelection(){if(!$('adaptiveScan').checked)return;const elapsed=Date.now()-highlightStarted,current=efficiency.scanMs||Number($('scanSpeed').value||2000);const ratio=elapsed/current;efficiency.timings=(efficiency.timings||[]).concat([ratio]).slice(-8);const avg=efficiency.timings.reduce((a,b)=>a+b,0)/efficiency.timings.length;let next=current;if(efficiency.timings.length>=3){if(avg>0.72)next=Math.min(3000,current+100);else if(avg<0.38)next=Math.max(1400,current-100)}efficiency.scanMs=next;saveEfficiency()}
function beginPhraseJourney(){if(phraseJourneyStart===null){phraseJourneyStart=Date.now();phraseSelections=0}}
function countSelection(){phraseSelections++;adaptScanFromSelection()}
function completePhraseMetric(){if(phraseJourneyStart===null)phraseJourneyStart=Date.now();efficiency.phrases=(efficiency.phrases||0)+1;efficiency.totalMs=(efficiency.totalMs||0)+(Date.now()-phraseJourneyStart);efficiency.totalSelections=(efficiency.totalSelections||0)+Math.max(1,phraseSelections);phraseJourneyStart=null;phraseSelections=0;saveEfficiency()}
function setMode(next){mode=next;const sw=mode==='switch';$('directMode').classList.toggle('active',!sw);$('switchMode').classList.toggle('active',sw);$('switchPanel').classList.toggle('hidden',!sw);$('modeTitle').textContent=sw?'One-movement switch access':'Direct gesture-to-speech';$('modeDesc').textContent=sw?'Auto-scan highlights choices. Raise either hand above your shoulder to SELECT.':'Best when the user can reliably produce several distinct movements.';$('liveEyebrow').textContent=sw?'LEARNED MOVEMENT = SELECT':'LIVE COMMUNICATION';$('chainHint').textContent=sw?'Your selected movement acts like an accessibility switch.':'Tip: teach WANT, NEED, CALL, or PLEASE to chain gestures into short phrases.';$('spokenPhrase').textContent=sw?'One movement can unlock a full vocabulary.':'Your movement becomes speech.';if(sw){renderBoard('HOME');startScan()}else stopScan()}
function renderBoard(name){currentBoard=name;scanIndex=0;$('boardBreadcrumb').textContent=name;const items=BOARDS[name]||BOARDS.HOME;$('phraseBoard').innerHTML=items.map((x,i)=>`<button class="phrase-tile ${i===0?'active':''}" data-i="${i}"><span>${x[1]}</span><strong>${escapeHtml(x[2])}</strong></button>`).join('');document.querySelectorAll('.phrase-tile').forEach(b=>b.onclick=()=>{scanIndex=Number(b.dataset.i);paintScan();selectHighlighted()});$('boardBack').style.visibility=name==='HOME'?'hidden':'visible';noteHighlight()}
function paintScan(){const tiles=[...document.querySelectorAll('.phrase-tile')];tiles.forEach((t,i)=>t.classList.toggle('active',i===scanIndex));if(tiles[scanIndex])$('boardStatus').textContent=`Ready: ${tiles[scanIndex].innerText.trim().replace(/\n/g,' ')}`;noteHighlight()}
function startScan(){stopScan();const speed=$('adaptiveScan').checked?(efficiency.scanMs||2000):Number($('scanSpeed').value||2000);scanTimer=setInterval(()=>{const tiles=[...document.querySelectorAll('.phrase-tile')];if(!tiles.length)return;scanIndex=(scanIndex+1)%tiles.length;paintScan()},speed);$('adaptiveScanState').textContent=`${speed} ms`}
function stopScan(){if(scanTimer){clearInterval(scanTimer);scanTimer=null}}
function selectHighlighted(){const item=(BOARDS[currentBoard]||[])[scanIndex];if(!item)return;beginPhraseJourney();countSelection();if(item[0]==='SAY'){speakBoard(item[2],false)}else if(item[0]==='HELP_NOW'){speakBoard('I need help now.',true)}else{renderBoard(item[0]);startScan()}}
function speakBoard(text,urgent){$('spokenPhrase').textContent='“'+text+'”';$('boardStatus').textContent='Spoken: '+text;speak(text,urgent);if(urgent)flashEmergency();completePhraseMetric();setTimeout(()=>{renderBoard('HOME');startScan()},urgent?2200:1500)}
$('autoRestBtn').onclick=async()=>{try{$('progressFill').style.width='0%';$('label').value='REST';$('phrase').value='Rest / no intentional gesture';$('emergency').checked=false;await api('/api/auto-rest',{});$('recordStatus').textContent='Auto REST started — sit naturally and remain comfortably still.';$('judgeStatus').textContent='Capturing 8 REST examples automatically. Breathe and blink normally.'}catch(e){$('judgeStatus').textContent=e.message}};
$('judgeTeachBtn').onclick=()=>{$('label').value='SELECT';$('phrase').value='Select';$('emergency').checked=false;document.querySelector('.teach').scrollIntoView({behavior:'smooth',block:'center'});$('recordStatus').textContent='Press Auto-record once, then perform SELECT 8 times. Return to REST between repetitions.';};
$('judgeTrainBtn').onclick=async()=>{try{$('judgeStatus').textContent='Training personalized model…';const j=await api('/api/train',{});await loadProfile();setMode('switch');$('selectGesture').value='SELECT';$('judgeStatus').textContent=`Ready. Validation ${Math.round(j.accuracy*100)}%. Use SELECT to control the phrase board.`;document.getElementById('switchPanel').scrollIntoView({behavior:'smooth',block:'start'})}catch(e){$('judgeStatus').textContent=e.message}};
$('recordBtn').onclick=async()=>{try{$('progressFill').style.width='0%';await api('/api/start-recording',{label:$('label').value,phrase:$('phrase').value,emergency:$('emergency').checked});$('recordStatus').textContent='Auto-recording 8 examples. Perform the gesture, return to REST during each short pause, then repeat.'}catch(e){$('recordStatus').textContent=e.message}}
$('trainBtn').onclick=async()=>{try{$('trainBtn').disabled=true;$('trainStatus').textContent='Training…';const j=await api('/api/train',{});$('trainStatus').textContent=`Cross-validated accuracy: ${Math.round(j.accuracy*100)}% · ${j.samples} samples`;await loadProfile()}catch(e){$('trainStatus').textContent=e.message}finally{$('trainBtn').disabled=false}}
$('speakToggle').onclick=()=>{speechOn=!speechOn;$('speakToggle').textContent=speechOn?'🔊 Speech on':'🔇 Speech off'}
$('directMode').onclick=()=>setMode('direct');$('switchMode').onclick=()=>setMode('switch');$('manualSelect').onclick=selectHighlighted;$('boardBack').onclick=()=>{phraseJourneyStart=null;phraseSelections=0;renderBoard('HOME');startScan()};$('scanSpeed').onchange=()=>{if(!$('adaptiveScan').checked){efficiency.scanMs=Number($('scanSpeed').value);saveEfficiency()}startScan()};$('adaptiveScan').onchange=()=>{renderEfficiency();startScan()};$('resetEfficiency').onclick=()=>{Object.assign(efficiency,{phrases:0,totalMs:0,totalSelections:0,scanMs:2000,timings:[]});saveEfficiency();startScan()};
document.querySelectorAll('[data-preset]').forEach(b=>b.onclick=()=>{const parts=b.dataset.preset.split('|');$('label').value=parts[0];$('phrase').value=parts[1]||parts[0];$('emergency').checked=parts.includes('emergency')})
$('resetBtn').onclick=async()=>{if(confirm('Delete all learned gestures and demo history?')){await api('/api/reset',{});await loadProfile();$('gestureName').textContent='Waiting for a gesture';$('spokenPhrase').textContent='Your movement becomes speech.'}}


async function recordTiny(label){try{$('movementProgress').style.width='0%';await api('/api/movement/start-recording',{label});$('movementStatus').textContent=label==='NEUTRAL'?'Recording neutral: stay comfortably still.':'Recording DISCOVER: make the same tiny comfortable movement now.'}catch(e){$('movementStatus').textContent=e.message}}
$('tinyNeutralBtn').onclick=()=>recordTiny('NEUTRAL');
$('tinyDiscoverBtn').onclick=()=>recordTiny('DISCOVER');
$('tinyAnalyzeBtn').onclick=async()=>{try{$('tinyAnalyzeBtn').disabled=true;$('movementStatus').textContent='Searching body signals for the cleanest tiny control…';const j=await api('/api/movement/discover-tiny',{});$('movementStatus').textContent=`Discovered: ${j.label} ${j.direction_word} · ${Math.round(j.score)} / 100`;await loadProfile()}catch(e){$('movementStatus').textContent=e.message}finally{$('tinyAnalyzeBtn').disabled=false}};
$('tinyEnableBtn').onclick=()=>{$('useTinyControl').checked=true;localStorage.setItem('gvUseTiny','1');setMode('switch');$('boardStatus').textContent='Auto-discovered tiny control is now SELECT.';$('spokenPhrase').textContent='Tiny movement enabled → SELECT';};
$('useTinyControl').checked=localStorage.getItem('gvUseTiny')==='1';
$('useTinyControl').onchange=()=>{localStorage.setItem('gvUseTiny',$('useTinyControl').checked?'1':'0');$('boardStatus').textContent=$('useTinyControl').checked?'Tiny auto-discovered control enabled.':'Learned gesture control enabled.'};

document.querySelectorAll('[data-move]').forEach(b=>b.onclick=()=>{$('movementLabel').value=b.dataset.move;$('movementStatus').textContent=b.dataset.move==='NEUTRAL'?'Remain comfortably still during the sample.':'Perform the same comfortable movement each time.'})
$('movementRecordBtn').onclick=async()=>{try{$('movementProgress').style.width='0%';await api('/api/movement/start-recording',{label:$('movementLabel').value});$('movementStatus').textContent='Get ready… recording starts with the live camera frames.'}catch(e){$('movementStatus').textContent=e.message}}
$('movementAnalyzeBtn').onclick=async()=>{try{$('movementAnalyzeBtn').disabled=true;$('movementStatus').textContent='Training movement reliability models…';const j=await api('/api/movement/analyze',{});$('movementStatus').textContent=`Best control: ${j.recommended} · ${j.movements[j.recommended].score}/100`;await loadProfile()}catch(e){$('movementStatus').textContent=e.message}finally{$('movementAnalyzeBtn').disabled=false}}
$('movementResetBtn').onclick=async()=>{if(confirm('Reset only the movement reliability calibration?')){await api('/api/movement/reset',{});$('movementStatus').textContent='Movement lab reset.';await loadProfile()}}
$('driftResetBtn').onclick=async()=>{await api('/api/drift/reset',{});$('driftState').classList.remove('adapting');$('driftState').innerHTML='<small>ADAPTATION STATUS</small><strong>Drift observations reset</strong><span>Training baselines are preserved.</span>';$('currentBar').style.width='0%';$('strengthPercent').textContent='100%';$('driftHistory').innerHTML='<div class="empty">No adaptive observations yet.</div>';await loadProfile()};
renderEfficiency();loadProfile();startCamera();

// v11 presentation mode
(function(){
  const btn=document.getElementById('presentationToggle');
  if(!btn) return;
  function apply(on){
    document.body.classList.toggle('presentation-mode',on);
    btn.classList.toggle('active',on);
    btn.textContent=on?'Exit presentation':'Presentation mode';
    try{localStorage.setItem('gesturevoice_presentation',on?'1':'0')}catch(e){}
  }
  btn.addEventListener('click',()=>apply(!document.body.classList.contains('presentation-mode')));
  let saved=false;try{saved=localStorage.getItem('gesturevoice_presentation')==='1'}catch(e){}
  apply(saved);
})();

if($('useHandsUpControl'))$('useHandsUpControl').onchange=()=>{
 handsUpSince=null;handsUpLatched=false;handsDownSince=Date.now();
 if($('useHandsUpControl').checked){$('useTinyControl').checked=false;$('boardStatus').textContent='Hands-up switch ready. Raise either hand above your shoulder to select.';}
};
