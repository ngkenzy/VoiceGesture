from __future__ import annotations
import numpy as np

# Body order from vision.py: nose, LS, RS, LE, RE, LW, RW, LH, RH. Each point = x,y,z,visibility.

def _resample(seq, target=18):
    a=np.asarray(seq,dtype=np.float32)
    if a.ndim!=2 or len(a)<4: raise ValueError('Not enough frames')
    if len(a)==target:return a
    old=np.linspace(0,1,len(a)); new=np.linspace(0,1,target)
    return np.vstack([np.interp(new,old,a[:,i]) for i in range(a.shape[1])]).T.astype(np.float32)

def _safe_angle(v1,v2):
    n1=np.linalg.norm(v1,axis=-1); n2=np.linalg.norm(v2,axis=-1)
    den=np.maximum(n1*n2,1e-6)
    c=np.clip(np.sum(v1*v2,axis=-1)/den,-1,1)
    return np.arccos(c)

def kinematic_sequence(seq, target=18):
    """Return compact causal per-frame movement features.

    Includes normalized positions, velocity, acceleration and interpretable joint/torso angles.
    All values are based on body-normalized landmarks already produced by vision.py.
    """
    a=_resample(seq,target)
    if a.shape[1] < 36: raise ValueError('Pose features unavailable')
    body=a[:,:36].reshape(target,9,4)
    xy=body[:,:,:2]
    vis=body[:,:,3]
    vel=np.diff(xy,axis=0,prepend=xy[:1])
    acc=np.diff(vel,axis=0,prepend=vel[:1])

    nose,ls,rs,le,re,lw,rw,lh,rh=[xy[:,i] for i in range(9)]
    shoulder_mid=(ls+rs)/2
    hip_mid=(lh+rh)/2
    shoulder_vec=rs-ls
    torso_vec=shoulder_mid-hip_mid
    head_vec=nose-shoulder_mid
    head_tilt=np.arctan2(head_vec[:,0],np.maximum(np.abs(head_vec[:,1]),1e-5))[:,None]
    shoulder_tilt=np.arctan2(shoulder_vec[:,1],np.maximum(np.abs(shoulder_vec[:,0]),1e-5))[:,None]
    torso_tilt=np.arctan2(torso_vec[:,0],np.maximum(np.abs(torso_vec[:,1]),1e-5))[:,None]
    l_elbow=_safe_angle(ls-le,lw-le)[:,None]
    r_elbow=_safe_angle(rs-re,rw-re)[:,None]
    wrist_sep=np.linalg.norm(lw-rw,axis=1)[:,None]
    wrist_speed=np.stack([np.linalg.norm(vel[:,5],axis=1),np.linalg.norm(vel[:,6],axis=1)],axis=1)
    wrist_acc=np.stack([np.linalg.norm(acc[:,5],axis=1),np.linalg.norm(acc[:,6],axis=1)],axis=1)

    # Positions of upper body (nose through wrists only) + velocity/acceleration + angles.
    pos=xy[:,:7].reshape(target,-1)
    vv=vel[:,:7].reshape(target,-1)
    aa=acc[:,:7].reshape(target,-1)
    visibility=vis[:,:7]
    return np.concatenate([pos,vv,aa,visibility,head_tilt,shoulder_tilt,torso_tilt,l_elbow,r_elbow,wrist_sep,wrist_speed,wrist_acc],axis=1).astype(np.float32)

def kinematic_summary(seq):
    k=kinematic_sequence(seq)
    d=np.diff(k,axis=0)
    # Compact fixed vector used by the fusion model.
    return np.concatenate([
        k.mean(0),k.std(0),k.min(0),k.max(0),k[-1]-k[0],
        np.abs(d).mean(0),np.abs(d).max(0)
    ]).astype(np.float32)

def movement_metrics(seq):
    k=kinematic_sequence(seq)
    # Last columns: head, shoulder, torso, elbows, wrist sep, 2 speeds, 2 accels.
    head=np.degrees(k[:,-11])
    wrist_speed=k[:,-4:-2]
    wrist_acc=k[:,-2:]
    jerk=np.diff(wrist_acc,axis=0)
    return {
        'head_range_deg': float(np.ptp(head)),
        'peak_wrist_speed': float(np.max(wrist_speed)),
        'mean_wrist_speed': float(np.mean(wrist_speed)),
        'peak_wrist_acceleration': float(np.max(wrist_acc)),
        'smoothness': float(1.0/(1.0+np.mean(np.abs(jerk)))) if len(jerk) else 1.0,
    }
