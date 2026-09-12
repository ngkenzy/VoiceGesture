from __future__ import annotations
from pathlib import Path
import base64
import os
import re
import requests
from flask import Response
import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request

from vision import VisionPipeline
from gesture_engine import GestureEngine
from movement_reliability import MovementReliabilityEngine

app = Flask(__name__)
engine = GestureEngine(window=18)
movement = MovementReliabilityEngine(window=18)
vision = VisionPipeline()


ELEVENLABS_KEY_FILE = os.path.join(os.path.dirname(__file__), 'ElevenlabsAPI.txt')
ELEVENLABS_DEFAULT_VOICE = os.environ.get('ELEVENLABS_VOICE_ID', 'JBFqnCBsd6RMkjVDRZzb')  # George; current docs example voice
ELEVENLABS_MODEL = os.environ.get('ELEVENLABS_MODEL', 'eleven_flash_v2_5')


def load_elevenlabs_key():
    """Read the ElevenLabs key from ElevenlabsAPI.txt/RTF without ever logging it."""
    try:
        raw = Path(ELEVENLABS_KEY_FILE).read_text(errors='ignore')
    except Exception:
        return None
    # Accept plain text or an RTF-wrapped key.
    m = re.search(r'(?<![A-Za-z0-9])sk_[A-Za-z0-9_-]{20,}', raw)
    return m.group(0) if m else None


def elevenlabs_enabled():
    return bool(load_elevenlabs_key())


def _elevenlabs_error(r):
    """Return a sanitized ElevenLabs error without ever exposing credentials."""
    detail = ''
    status = ''
    try:
        body = r.json()
        d = body.get('detail')
        if isinstance(d, dict):
            status = str(d.get('status') or '')
            detail = str(d.get('message') or d.get('status') or '')
        elif isinstance(d, str):
            detail = d
    except Exception:
        pass
    return status, detail or f'ElevenLabs returned HTTP {r.status_code}'


def elevenlabs_probe():
    """Check local configuration only.

    Do NOT call /v1/voices here: ElevenLabs API keys are scope-restricted by
    default, and a key can have text_to_speech permission without voice-list
    permission. The real TTS endpoint is the authoritative connectivity test.
    """
    key = load_elevenlabs_key()
    if not key:
        return {'ok': False, 'stage': 'key', 'error': 'API key not found in ElevenlabsAPI.txt'}
    voice_id = os.environ.get('ELEVENLABS_VOICE_ID', ELEVENLABS_DEFAULT_VOICE)
    return {
        'ok': True,
        'stage': 'configured',
        'voice_id': voice_id,
        'model': ELEVENLABS_MODEL,
        'message': 'API key loaded. Use Test ElevenLabs to verify TTS permission and quota.'
    }


def decode_data_url(data_url):
    raw = data_url.split(',',1)[-1]
    arr = np.frombuffer(base64.b64decode(raw), np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)

def encode_jpg(frame):
    ok, buf = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 78])
    if not ok: return None
    return 'data:image/jpeg;base64,' + base64.b64encode(buf).decode('ascii')

@app.get('/')
def home():
    return render_template('index.html')

@app.get('/api/profile')
def profile():
    p = engine.profile()
    p['yolo11_enabled'] = vision.yolo11_pose.enabled
    p['yolo_model'] = vision.yolo11_pose.model_name
    p['movement_lab'] = movement.profile()
    p['pose_backend'] = vision.pose_backend_status
    p['elevenlabs_enabled'] = elevenlabs_enabled()
    p['elevenlabs_model'] = ELEVENLABS_MODEL if p['elevenlabs_enabled'] else None
    return jsonify(p)

@app.post('/api/start-recording')
def start_recording():
    d = request.get_json(force=True)
    try:
        return jsonify({'ok':True, **engine.start_recording(d.get('label',''), d.get('phrase',''), bool(d.get('emergency', False)))})
    except Exception as e:
        return jsonify({'ok':False,'error':str(e)}), 400

@app.post('/api/auto-rest')
def auto_rest():
    try:
        return jsonify({'ok':True, **engine.start_auto_rest(8)})
    except Exception as e:
        return jsonify({'ok':False,'error':str(e)}), 400

@app.post('/api/cancel-recording')
def cancel_recording():
    engine.cancel_recording(); return jsonify({'ok':True})

@app.post('/api/train')
def train():
    try:
        return jsonify({'ok':True, **engine.train()})
    except Exception as e:
        return jsonify({'ok':False,'error':str(e)}), 400


@app.post('/api/movement/start-recording')
def movement_start_recording():
    d = request.get_json(force=True)
    try:
        return jsonify({'ok':True, **movement.start_recording(d.get('label',''))})
    except Exception as e:
        return jsonify({'ok':False,'error':str(e)}), 400

@app.post('/api/movement/analyze')
def movement_analyze():
    try:
        return jsonify({'ok':True, **movement.analyze()})
    except Exception as e:
        return jsonify({'ok':False,'error':str(e)}), 400

@app.post('/api/movement/discover-tiny')
def movement_discover_tiny():
    try:
        return jsonify({'ok': True, **movement.analyze_tiny()})
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)}), 400

@app.post('/api/movement/reset-tiny')
def movement_reset_tiny():
    movement.reset_tiny()
    return jsonify({'ok': True})

@app.post('/api/movement/reset')
def movement_reset():
    movement.reset()
    return jsonify({'ok':True})


@app.post('/api/drift/reset')
def drift_reset():
    engine.reset_drift()
    return jsonify({'ok':True})

@app.post('/api/reset')
def reset():
    engine.reset_all(); movement.reset(); return jsonify({'ok':True})



@app.get('/api/tts/status')
def elevenlabs_status():
    return jsonify(elevenlabs_probe())


@app.post('/api/tts')
def elevenlabs_tts():
    d = request.get_json(force=True)
    text = (d.get('text') or '').strip()
    urgent = bool(d.get('urgent', False))
    if not text:
        return jsonify({'ok': False, 'error': 'No text provided.'}), 400
    if len(text) > 500:
        return jsonify({'ok': False, 'error': 'Text is too long.'}), 400

    key = load_elevenlabs_key()
    if not key:
        return jsonify({'ok': False, 'error': 'ElevenLabs API key file not found.'}), 503

    # Call Text-to-Speech directly. Do not preflight /v1/voices because API keys
    # may legitimately have text_to_speech permission while voice-list access is restricted.
    voice_id = os.environ.get('ELEVENLABS_VOICE_ID', ELEVENLABS_DEFAULT_VOICE)

    url = f'https://api.elevenlabs.io/v1/text-to-speech/{voice_id}'
    # Keep the request deliberately minimal for maximum model/account compatibility.
    payload = {
        'text': text,
        'model_id': ELEVENLABS_MODEL,
        'voice_settings': {
            'stability': 0.45 if urgent else 0.60,
            'similarity_boost': 0.78,
        }
    }
    headers = {'xi-api-key': key, 'Content-Type': 'application/json', 'Accept': 'audio/mpeg'}
    try:
        r = requests.post(url, headers=headers, json=payload,
                          params={'output_format': 'mp3_44100_128'}, timeout=20)
        if r.status_code != 200:
            detail = ''
            try:
                body = r.json()
                d = body.get('detail')
                if isinstance(d, dict): detail = d.get('message') or d.get('status') or ''
                elif isinstance(d, str): detail = d
            except Exception:
                pass
            status, message = _elevenlabs_error(r)
            return jsonify({'ok': False, 'provider': 'elevenlabs',
                            'error': message,
                            'elevenlabs_status': status,
                            'http_status': r.status_code,
                            'hint': ('Enable Text to Speech permission for this API key in ElevenLabs Developers → API Keys.'
                                     if status in ('missing_permissions','invalid_api_key') or r.status_code in (401,403)
                                     else 'Check your ElevenLabs quota, voice ID, and model access.')}), 502
        return Response(r.content, mimetype='audio/mpeg', headers={
            'Cache-Control': 'no-store',
            'X-Voice-Provider': 'ElevenLabs',
            'X-ElevenLabs-Voice-Id': voice_id,
        })
    except requests.RequestException:
        return jsonify({'ok': False, 'provider': 'elevenlabs', 'error': 'ElevenLabs request failed (network/timeout).'}), 502


@app.post('/api/frame')
def frame():
    d = request.get_json(force=True)
    img = decode_data_url(d['image'])
    feature, annotated, status = vision.extract(img)
    event = engine.add_frame(feature)
    move_event = movement.add_frame(feature)
    tiny = None if movement.recording else movement.tiny_live(feature)
    mode = d.get('mode','direct')
    pred = None if (engine.recording or movement.recording) else engine.infer(mode=mode)
    refresh = (event and event.get('sample_saved')) or (move_event and move_event.get('sample_saved'))
    return jsonify({'ok':True,'image':encode_jpg(annotated),'status':status,'event':event,'movement_event':move_event,
                    'prediction':pred,'tiny_prediction':tiny,'profile':profile().json if refresh else None})

if __name__ == '__main__':
    print('\nGestureVoice running at http://127.0.0.1:7860\n')
    app.run(host='127.0.0.1', port=7860, debug=False, threaded=False)
