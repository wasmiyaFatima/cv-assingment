
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import mediapipe as mp


VIDEO_PATH = "ModelVideo.mp4"
MODEL_PATH = "pose_landmarker_full.task"

SMOOTH_WIN = 7
KNEE_THRESH = 150


# 3. LOAD MEDIAPIPE MODEL

BaseOptions = python.BaseOptions
PoseLandmarker = vision.PoseLandmarker
PoseLandmarkerOptions = vision.PoseLandmarkerOptions
VisionRunningMode = vision.RunningMode

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.IMAGE
)

landmarker = PoseLandmarker.create_from_options(options)

# 4. LOAD VIDEO

cap = cv2.VideoCapture(VIDEO_PATH)

frames = []
keypoints = []
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frames.append(frame)
    frame_count += 1
    if frame_count % 100 == 0:
        print(f"Loading frames... {frame_count} processed")
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
    image_format=mp.ImageFormat.SRGB,
    data=rgb
    )

    result = landmarker.detect(mp_image)

    frame_kps = {}

    if result.pose_landmarks:
        for i, lm in enumerate(result.pose_landmarks[0]):
            x = int(lm.x * frame.shape[1])
            y = int(lm.y * frame.shape[0])
            frame_kps[i] = (x, y)

    keypoints.append(frame_kps)

cap.release()

print("Keypoints extracted using MediaPipe")


# 5. SMOOTHING

print(f"Smoothing {len(keypoints)} frames...")
smoothed = [{} for _ in range(len(keypoints))]

for idx in range(33):  # 33 landmarks
    if idx % 11 == 0:
        print(f"Smoothing landmark {idx}/33...")
    xs = np.array([keypoints[f].get(idx, (0,0))[0] for f in range(len(keypoints))])
    ys = np.array([keypoints[f].get(idx, (0,0))[1] for f in range(len(keypoints))])

    xs_s = uniform_filter1d(xs, size=SMOOTH_WIN)
    ys_s = uniform_filter1d(ys, size=SMOOTH_WIN)

    for f in range(len(keypoints)):
        smoothed[f][idx] = (xs_s[f], ys_s[f])

print(" Smoothing applied")


# 6. ANGLE FUNCTION

def angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cos = np.dot(ba, bc) / (np.linalg.norm(ba)*np.linalg.norm(bc) + 1e-9)
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))

# Landmark indices (MediaPipe)
LEFT_HIP, LEFT_KNEE, LEFT_ANKLE = 23, 25, 27
RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE = 24, 26, 28


# 7. COMPUTE ANGLES

print("Computing angles...")
left_knee = []
right_knee = []

for f in range(len(smoothed)):
    if f % 100 == 0:
        print(f"Angle computation... {f}/{len(smoothed)}")
    kps = smoothed[f]

    lk = angle(kps[LEFT_HIP], kps[LEFT_KNEE], kps[LEFT_ANKLE])
    rk = angle(kps[RIGHT_HIP], kps[RIGHT_KNEE], kps[RIGHT_ANKLE])

    left_knee.append(lk)
    right_knee.append(rk)

print("Angles computed")


#  CLASSIFICATION

print("Classifying frames...")
predictions = []

for i in range(len(left_knee)):
    if i % 100 == 0:
        print(f"Classification... {i}/{len(left_knee)}")
    avg_knee = (left_knee[i] + right_knee[i]) / 2

    if avg_knee > KNEE_THRESH:
        predictions.append("STANDING")
    else:
        predictions.append("SITTING")


# GROUND TRUTH (EDIT IF NEEDED)

ground_truth = ["STANDING" if i < len(predictions)//2 else "SITTING"
                for i in range(len(predictions))]


#  ACCURACY SCORES

correct = sum(p == g for p, g in zip(predictions, ground_truth))
accuracy = correct / len(predictions) * 100

print(f" Accuracy: {accuracy:.2f}%")


#  PLOT ANGLES

# Save plot instead of showing it
plt.figure(figsize=(12,5))
plt.plot(left_knee, label="Left Knee")
plt.plot(right_knee, label="Right Knee")
plt.axhline(KNEE_THRESH, color='r', linestyle='--', label="Threshold")
plt.legend()
plt.title("Knee Angle Over Time")
plt.xlabel("Frame")
plt.ylabel("Angle")
plt.savefig('knee_angles.png', dpi=100, bbox_inches='tight')
print("Plot saved as knee_angles.png")
plt.close()


# SKELETON VISUALIZATION 

import os
current_dir = os.getcwd()
output_path = os.path.join(current_dir, 'skeleton_overlay.avi')
print(f"Generating skeleton overlay video...")
print(f"Will save to: {output_path}")

# Get video properties for output
frame_width = 640
frame_height = 480
fps = 30  # frames per second
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

frame_indices = list(range(len(frames)))
for idx, i in enumerate(frame_indices):
    frame = frames[i].copy()
    kps = smoothed[i]

    # Draw lines 
    connections = [
        (23,25), (25,27),   # Left leg
        (24,26), (26,28),   # Right leg
        (23,24),            # Hips
        (11,23), (12,24),   # Body
        (11,12),            # Shoulders
        (11,13), (13,15),   # Left arm
        (12,14), (14,16)    # Right arm
    ]

    for a, b in connections:
        if a in kps and b in kps:
            x1, y1 = int(kps[a][0]), int(kps[a][1])
            x2, y2 = int(kps[b][0]), int(kps[b][1])
            cv2.line(frame, (x1,y1), (x2,y2), (0,255,0), 3)

    # keep circles too 
    for (x,y) in kps.values():
        cv2.circle(frame, (int(x),int(y)), 4, (0,0,255), -1)

    cv2.putText(frame, predictions[i], (20,40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

    frame = cv2.resize(frame, (frame_width, frame_height))
    out.write(frame)
    if idx % 50 == 0:
        print(f"Rendering video... {idx}/{len(frame_indices)} frames")

out.release()
print(f"Video saved to: {output_path}")