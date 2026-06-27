# REDRED — Gesture Recognition Camera

A real-time webcam app that recognizes static hand gestures with
[MediaPipe](https://ai.google.dev/edge/mediapipe) hand landmarks and tints the
screen accordingly.

There are two gestures. When you hold up both hands as open palms facing the
camera, the screen turns red at 50% transparency and a large `RED RED` label
appears in the center. When you make the metal (rock) sign with both hands,
raising the index and pinky fingers while folding the middle and ring fingers,
the screen turns green at 50% transparency with a large `GREEN GREEN` label.

While your hands are tracked, they are drawn on screen as a neon-green skeleton
with yellow joints. Press **ESC** or **q** at any time to quit.

## How it works

- **MediaPipe HandLandmarker** (a pre-trained model) finds 21 landmarks per hand each frame.
- Gestures are detected with simple **geometric rules** on those landmarks
  (no training required). Palm-vs-back-of-hand is told apart using the palm
  "winding" (a 2D cross product) combined with handedness.

## Requirements

- Python 3.11+ (developed on 3.14)
- A webcam
- Packages in [`requirements.txt`](requirements.txt): `mediapipe`, `opencv-contrib-python`, `numpy`
- The model file [`hand_landmarker.task`](hand_landmarker.task) (included in this repo)

## Setup

### Option A — using [uv](https://docs.astral.sh/uv/) (recommended)

```bash
uv venv
uv pip install -r requirements.txt
```

### Option B — using pip + venv

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

If you don't have the model file, download it once:

```bash
curl -L -o hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
```

## Run

```bash
# with the venv activated
python Redgreen.py

# or without activating
.venv/Scripts/python.exe Redgreen.py   # Windows
.venv/bin/python Redgreen.py           # macOS / Linux
```

