from __future__ import annotations
from pathlib import Path
import json
import numpy as np
from kinematics import kinematic_sequence

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE=True
except Exception:
    torch=None; nn=None; TORCH_AVAILABLE=False

if TORCH_AVAILABLE:
    class TinyCausalGRU(nn.Module):
        def __init__(self,input_dim,hidden_dim,num_classes):
            super().__init__()
            self.gru=nn.GRU(input_dim,hidden_dim,batch_first=True)
            self.head=nn.Sequential(nn.LayerNorm(hidden_dim),nn.Linear(hidden_dim,num_classes))
        def forward(self,x):
            y,_=self.gru(x)
            return self.head(y[:,-1,:])

class GRUGestureModel:
    def __init__(self, model_dir='models'):
        self.model_dir=Path(model_dir); self.model_dir.mkdir(parents=True,exist_ok=True)
        self.path=self.model_dir/'causal_gru.pt'; self.meta_path=self.model_dir/'causal_gru.json'
        self.model=None; self.classes=[]; self.input_dim=None
        self.available=TORCH_AVAILABLE
        if self.available:self._load()

    def _load(self):
        if not (self.path.exists() and self.meta_path.exists()):return
        try:
            meta=json.loads(self.meta_path.read_text()); self.classes=meta['classes']; self.input_dim=int(meta['input_dim'])
            self.model=TinyCausalGRU(self.input_dim,int(meta.get('hidden_dim',48)),len(self.classes))
            self.model.load_state_dict(torch.load(self.path,map_location='cpu',weights_only=True)); self.model.eval()
        except Exception:
            self.model=None

    def fit(self,sequences,labels,epochs=70):
        if not self.available:return {'available':False,'trained':False,'reason':'PyTorch not installed'}
        classes=sorted(set(map(str,labels)))
        if len(classes)<2:return {'available':True,'trained':False,'reason':'Need 2+ classes'}
        X=np.stack([kinematic_sequence(s) for s in sequences]).astype(np.float32)
        y=np.asarray([classes.index(str(v)) for v in labels],dtype=np.int64)
        self.input_dim=X.shape[-1]; self.classes=classes
        model=TinyCausalGRU(self.input_dim,48,len(classes))
        opt=torch.optim.Adam(model.parameters(),lr=0.005,weight_decay=1e-4)
        loss_fn=nn.CrossEntropyLoss()
        xt=torch.from_numpy(X); yt=torch.from_numpy(y)
        model.train()
        for _ in range(int(epochs)):
            # light jitter improves tolerance to tiny changes without inventing labels
            noise=torch.randn_like(xt)*0.006
            logits=model(xt+noise)
            loss=loss_fn(logits,yt)
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval(); self.model=model
        with torch.no_grad():
            probs=torch.softmax(model(xt),dim=1); pred=probs.argmax(1)
            train_acc=float((pred==yt).float().mean().item())
        torch.save(model.state_dict(),self.path)
        self.meta_path.write_text(json.dumps({'classes':classes,'input_dim':self.input_dim,'hidden_dim':48},indent=2))
        return {'available':True,'trained':True,'classes':classes,'training_accuracy':train_acc,'samples':len(y)}

    def predict_proba(self,seq):
        if not (self.available and self.model is not None):return None
        X=kinematic_sequence(seq)[None,:,:]
        with torch.no_grad():
            p=torch.softmax(self.model(torch.from_numpy(X)),dim=1).numpy()[0]
        return {c:float(p[i]) for i,c in enumerate(self.classes)}

    def reset(self):
        self.model=None; self.classes=[]
        for p in [self.path,self.meta_path]:
            if p.exists():p.unlink()
