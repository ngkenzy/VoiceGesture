from pathlib import Path
import requests

URL='https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.onnx'
out=Path('models/yolox_s.onnx')
out.parent.mkdir(exist_ok=True)
if out.exists() and out.stat().st_size>1_000_000:
    print(f'YOLOX-S already present: {out}')
    raise SystemExit(0)
print('Downloading official YOLOX-S ONNX model…')
with requests.get(URL,stream=True,timeout=60) as r:
    r.raise_for_status()
    with out.open('wb') as f:
        for chunk in r.iter_content(1024*1024):
            if chunk: f.write(chunk)
print(f'Saved {out} ({out.stat().st_size/1e6:.1f} MB)')
