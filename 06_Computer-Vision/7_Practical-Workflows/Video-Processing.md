# Video Processing

> **Time is the 4th Dimension** - Optical Flow and Object Tracking

---

## 🌊 Optical Flow

Estimating the motion of pixels between two consecutive frames.
**Assumption:** Brightness constancy (pixel intensity doesn't change, it just moves).

$$ I(x, y, t) = I(x+dx, y+dy, t+dt) $$

### Lucas-Kanade Method (Sparse)
Tracks specific feature points (corners).
Fast but loses track if points move too fast.

### Farneback Method (Dense)
Calculates flow for *every* pixel.
Output: Vector field (Angle = Direction, Magnitude = Speed).

```python
# Dense Optical Flow
prev_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
# flow is (H, W, 2) -> (dx, dy)
```

---

## 🎯 Object Tracking (DeepSORT)

**Detection:** Finds objects in *one* frame.
**Tracking:** Connects objects across *many* frames (assigns ID #1 to the same car).

**DeepSORT Algorithm:**
1.  **Kalman Filter:** Predicts where the object *should* be in the next frame based on velocity.
2.  **Hungarian Algorithm:** Matches predicted boxes to detected boxes (IoU).
3.  **Deep Appearance Descriptor:** Uses a CNN (ReID model) to ensure the visual appearance matches (handles occlusion).

---

## 💻 Implementation (DeepSORT)

```python
from deep_sort_realtime.deepsort_tracker import DeepSort

tracker = DeepSort(max_age=30)

# In loop:
# detections = [[x1, y1, w, h, score], ...]
tracks = tracker.update_tracks(detections, frame=frame)

for track in tracks:
    if not track.is_confirmed():
        continue
    track_id = track.track_id
    ltrb = track.to_ltrb() # Left, Top, Right, Bottom
    # Draw ID
```

---

## 🎓 Interview Focus

1.  **Detection vs Tracking?**
    - Detection runs on every frame (expensive).
    - Tracking runs on every frame (cheap) and uses Detection only occasionally to correct drift (Detection-by-Tracking).

2.  **What is the Kalman Filter?**
    - A recursive mathematical tool that estimates the state of a system (position, velocity) from noisy measurements. It smooths out jittery detections.

3.  **Occlusion Handling?**
    - If a person walks behind a tree, the detector loses them. The Kalman filter predicts they are still moving. DeepSORT keeps the track "alive" (max_age) and re-identifies them when they reappear using the appearance embedding.

---

**Video: Giving AI a sense of time!**
