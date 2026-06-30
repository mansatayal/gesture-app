# Gesture App

A real-time hand gesture control app built with OpenCV and MediaPipe. Control presentations, draw in the air, and apply custom face filters — all using just your hands and a webcam.

## Features

### 🎨 Draw Mode
Air-draw on a virtual canvas using your index finger.
- **Index finger** — thin brush stroke
- **Index + middle finger (peace sign)** — thick brush stroke
- **Index + middle + pinky** — clear the entire canvas
- **Thumb + index pinch** — eraser
- **Thumb + index + middle spread** — resize the eraser on the fly
- **Color strip** (left side of screen) — hold your index finger over a swatch for 1 second to switch colors

### 🖥️ Count / Present Mode
Control a presentation (PowerPoint, Google Slides, etc.) hands-free.
- **Index finger** — next slide
- **Thumb** — previous slide
- **All five fingers** — start presentation (F5)
- **Fist** — stop presentation (Esc)

Each gesture requires a 1-second hold to trigger, with audio confirmation so you always know an action registered.

### 🙂 Filter Mode
Draw your own custom face filter, then watch it stick to your face in real time.
1. **Capture** — show your face to the camera; the app captures your facial landmarks using MediaPipe FaceMesh
2. **Template** — draw directly onto a skeleton outline of your own face (same drawing gestures as Draw Mode), placing eyebrows, glasses, a mustache, or anything else exactly where you want it
3. **Apply** — hit the on-screen button and your drawing maps onto your live face, tracking your movements and expressions in real time
4. **Edit** — jump back into Template mode anytime to tweak the design

## How It Works

The app uses MediaPipe's hand landmark model to track 21 points per hand and classify finger positions (open/closed) frame by frame. Gestures are recognized from these finger states and held for a short duration before triggering an action, preventing accidental triggers from quick hand movements.

Filter Mode goes a step further: each pixel you draw on the face template is matched to its nearest facial landmark and stored as an offset from that point. When applied, the app recalculates landmark positions on every frame and redraws your strokes relative to their anchor points — so your custom filter moves naturally with your face.

## Setup

**Requirements:**
- Python 3.x
- A working webcam

See `requirements.txt` for Python package dependencies.

**Install dependencies:**
```bash
pip install -r requirements.txt
```

**Run:**
```bash
python main.py
```

## Usage

- Switch between **Draw**, **Filter**, and **Count** modes by clicking the corresponding button in the on-screen menu bar (mouse click, top of the window)
- Press **Esc** to quit

## Tech Stack

- **OpenCV** — webcam capture, rendering, UI
- **MediaPipe** — hand landmark detection, face mesh detection
- **Pygame** — audio feedback for gestures
- **PyAutoGUI** — sends keypresses to control external presentation software
- **NumPy** — image array manipulation, masking, pixel-level operations

## Notes & Limitations

- Finger detection assumes the palm faces the camera; accuracy drops with the back of the hand visible
- Finger detection assumes an upright hand; tilting reduces accuracy
- Two consecutive identical gestures in Count mode won't re-trigger the same action without a gesture change in between
- The face skeleton in Filter Mode is built from your facial landmarks at the moment of capture — if your face is tilted when the template is generated, the skeleton (and your custom filter) will be tilted accordingly
- Drawings can be saved as a screenshot/photo, but there's no way to re-import a saved drawing back into the app to keep editing — it's a snapshot, not an editable project file

## Future Improvements

- Use index finger as a mouse pointer
- Proper save/load for drawings and filters (not just screenshots)
- Undo/redo support in Draw and Filter modes
- Shareable filter marketplace — artists could create and share (or sell) custom face filters for others to import and use
