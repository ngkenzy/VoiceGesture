<<<<<<< HEAD
# VoiceGesture
=======
# GestureVoice Presentation v11.4 — Hand Capture Fix

This build improves Switch Board hand detection by fusing YOLO11 Pose, MediaPipe Pose, and MediaPipe Hands. Raise either hand and hold briefly to SELECT; lower it to re-arm. The camera overlay now shows HAND DETECTED - SELECT READY when the switch sees your hand.

# YOLO11 UPGRADE

This build uses **YOLO11s-pose as the primary body pose model**. See `README_YOLO11_UPGRADE.md`.

# GestureVoice Adaptive v10 — Reliability Edition

GestureVoice turns personalized body movement into speech and switch-access communication. v9 focuses on recognition reliability rather than adding more complexity.

## What changed in v9

- **REST / NONE class**: teach the system what intentional non-gesture time looks like so it can refuse to speak.
- **50% confidence floor preserved**: stable recognition must still reach 50% fused confidence before speech.
- **Longer temporal stability**: 6 matching votes normally, 5 only when adaptive drift mode is active.
- **Landmark dead zones**: tiny body/hand jitter is suppressed before classification.
- **Automatic GRU gating**: the causal GRU stays off until every non-rest gesture has at least 20 examples. Small datasets use the more dependable RF + kinematic fusion.
- **Adaptive fusion weights**: RF dominates with small personalized datasets; GRU receives more weight only as the dataset grows.
- **Training Quality score**: combines validation accuracy, REST coverage, and per-gesture sample coverage.
- **Gesture separation warnings**: warns when two learned gestures are too similar and gives a 0–100 separation score.
- **Clear calibration targets**: 8 REST/NONE examples and 8 examples per intentional gesture are the standard target.

All v8 capabilities remain: tiny-movement auto-discovery, switch board, 44 phrases, movement reliability lab, adaptive drift, kinematics, optional YOLO11 Pose, and optional causal GRU.

## Recommended training protocol

For the most reliable demo:

1. Record `REST` **8 times**. Sit naturally and do not intentionally gesture. Include small normal posture changes.
2. Record each intentional gesture **8 times**.
3. Vary each gesture slightly: larger/smaller, faster/slower, and small seating-position changes.
4. Train the personalized model.
5. Read the **Reliability Edition** panel. If two gestures score under ~60 separation, make them more visually distinct or record more examples.
6. At the 8-example target, GestureVoice intentionally relies on RF + kinematics. The optional GRU remains gated until 20+ examples per non-rest gesture because it is less reliable on tiny datasets.

## Run on macOS — reliable/default mode

```bash
cd ~/Downloads/GestureVoice_Adaptive_v9
chmod +x run_mac.sh
./run_mac.sh
```

Open:

```text
http://127.0.0.1:7860
```

Allow camera access.

## Run with neural extras / YOLO11 Pose

```bash
cd ~/Downloads/GestureVoice_Adaptive_v9
chmod +x run_v9_neural_mac.sh
./run_v9_neural_mac.sh
```

The neural launcher installs the optional PyTorch and Ultralytics packages. v9 will still gate the GRU until there is enough training data.

## Best first reliability test

Train just these three classes:

- `REST` — 8 examples
- `WATER` — 8 examples
- `HELP` — 8 examples

Use very different WATER and HELP movements at first. Train, then watch the Reliability panel. Once that works consistently, add more gestures.

## Important

GestureVoice is a hackathon/prototype assistive communication system. It is not a medical device, diagnostic tool, or substitute for clinically validated AAC equipment.

## v9.1 False-Trigger Fix

This patched build adds a hard REST/neutral arming gate:

- Live speech is locked until a stable REST/NONE state is observed.
- A non-rest class cannot speak unless measured movement also exceeds the user's REST movement baseline.
- After any spoken phrase or switch event, GestureVoice disarms and requires a return to REST before another trigger.
- Old models that do not contain a REST/NONE/NEUTRAL class are blocked from live speech instead of guessing.
- Training now uses a simple 8-and-8 rule: 8 REST examples and 8 examples for every intentional gesture.

For a clean recalibration, record REST while sitting naturally still. Include normal blinking, breathing, and small posture jitter in those REST examples so the model learns what "doing nothing" really looks like.


## v10 — Gesture Quality Coach
- One-click Auto REST captures all 8 neutral examples continuously.
- Every gesture sample gets immediate strength / consistency / separation coaching.
- Warns when a gesture is too close to REST or too similar to another learned intent.
- 60-second Judge Mode guides REST → SELECT → Train → Switch Board.
- Keeps YOLO11s-pose primary, MediaPipe fallback, 50% confidence floor, and the false-trigger REST gate.

## ElevenLabs voice output (v10.1)

This build reads the ElevenLabs API key directly from `ElevenlabsAPI.txt` in the project folder. The file can be plain text or RTF-wrapped. The key is never shown in the UI or printed to the terminal.

Speech flow:

`recognized gesture -> confirmed phrase -> ElevenLabs Flash v2.5 -> natural voice`

If ElevenLabs is unavailable, GestureVoice automatically falls back to the browser/system voice so the demo continues.

Default voice: ElevenLabs Adam premade voice. You can override it without editing code:

```bash
export ELEVENLABS_VOICE_ID="your_voice_id"
./run_mac.sh
```

Security: `ElevenlabsAPI.txt` is listed in `.gitignore`. Do not commit or publish the file.

## ElevenLabs v10.3 direct-TTS fix

This build no longer requires the API key to have permission to list `/v1/voices`.
ElevenLabs keys are scope-restricted by default, so a Text-to-Speech-only key can now work normally.

1. Start the app.
2. Click **Test ElevenLabs Voice**.
3. If it fails, the status line now reports the sanitized ElevenLabs error and a likely fix.
4. In ElevenLabs, ensure the API key has **Text to Speech** permission and has remaining quota.

The default voice is the current ElevenLabs documentation example voice. Override it if needed:

```bash
export ELEVENLABS_VOICE_ID="your_voice_id"
./run_mac.sh
```

## v11.2 Hands-Up Switch reliability patch
- In Switch Board, **Raise hand = SELECT** is enabled by default.
- Either wrist held clearly above its corresponding shoulder for ~350 ms triggers one selection.
- The switch re-arms only after the hand returns below the shoulder for ~350 ms, preventing repeated selections while the hand stays raised.
- Default scan speed is now **2000 ms** for presentation reliability, with conservative adaptive limits of 1400–3000 ms.
- This direct pose switch bypasses the learned gesture classifier, so Switch Board can work even when personalized gesture classes overlap.
>>>>>>> 690e59a (Initial GestureVoice hackathon build)
