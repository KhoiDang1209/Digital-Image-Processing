import cv2
import torch
import numpy as np
import mediapipe as mp
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------
# Config
# ----------------------------
MAX_LEN = 20
INPUT_DIM = 227
CONF_THRESHOLD = 20.0
DEVICE = "cpu"

# ----------------------------
# Safe camera open
# ----------------------------
cap = None
for i in range(4):
    tmp = cv2.VideoCapture(i)
    if tmp.isOpened():
        cap = tmp
        print(f"✅ Using camera index {i}")
        break
if cap is None:
    raise RuntimeError("❌ No camera available")

# ----------------------------
# Model
# ----------------------------
class ASLModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.lstm = nn.LSTM(INPUT_DIM, 256, 2, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        h = torch.cat([h[-2], h[-1]], dim=1)
        return self.fc(h)

ckpt = torch.load("asl_lstm_keypoint_model.pth", map_location="cpu", weights_only=False)
labels = ckpt["labels"]

model = ASLModel(len(labels))
model.load_state_dict(ckpt["model"])
model.eval()

# ----------------------------
# MediaPipe
# ----------------------------
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ----------------------------
# Keypoint extraction (227 dims)
# ----------------------------
def extract_keypoints(res):
    keypoints = []

    if res.pose_landmarks:
        for lm in res.pose_landmarks.landmark:
            keypoints.extend([lm.x, lm.y, lm.z])
    else:
        keypoints.extend([0.0] * 33 * 3)

    if res.left_hand_landmarks:
        for lm in res.left_hand_landmarks.landmark:
            keypoints.extend([lm.x, lm.y, lm.z])
    else:
        keypoints.extend([0.0] * 21 * 3)

    if res.right_hand_landmarks:
        for lm in res.right_hand_landmarks.landmark:
            keypoints.extend([lm.x, lm.y, lm.z])
    else:
        keypoints.extend([0.0] * 21 * 3)

    keypoints.extend([
        float(res.left_hand_landmarks is not None),
        float(res.right_hand_landmarks is not None)
    ])

    return np.array(keypoints, dtype=np.float32)

# ----------------------------
# Webcam loop (fixed: no-hand reject)
# ----------------------------
sequence = []
NO_HAND_MAX = 5      # cho phép miss tay tối đa 5 frame
no_hand_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = holistic.process(img)

    kp = extract_keypoints(res)
    lh_present, rh_present = kp[-2], kp[-1]

    # ❗ Không có tay → reset sequence & không predict
    if lh_present == 0 and rh_present == 0:
        no_hand_count += 1
        if no_hand_count >= NO_HAND_MAX:
            sequence.clear()
            cv2.putText(
                frame, "No sign",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 0, 255), 2
            )
        cv2.imshow("ASL Recognition", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
        continue
    else:
        no_hand_count = 0

    sequence.append(kp)
    if len(sequence) > MAX_LEN:
        sequence = sequence[-MAX_LEN:]

    if len(sequence) == MAX_LEN:
        x = torch.from_numpy(np.array(sequence)).unsqueeze(0)
        with torch.no_grad():
            out = model(x)
            probs = F.softmax(out, dim=1)
            conf, idx = torch.max(probs, dim=1)

        confidence = conf.item() * 100
        if confidence >= CONF_THRESHOLD:
            pred = labels[idx.item()]
            cv2.putText(
                frame,
                f"{pred} ({confidence:.1f}%)",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2
            )
        else:
            cv2.putText(
                frame, "No sign",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 0, 255), 2
            )

    cv2.imshow("ASL Recognition", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()