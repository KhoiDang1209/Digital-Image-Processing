import os
import cv2
import torch
import numpy as np
import mediapipe as mp
import torch.nn as nn
import torch.nn.functional as F
from model import GRUClassifier

# ----------------------------
# Config
# ----------------------------
MAX_LEN = 20
# This GRU was trained on 63-dim one-hand keypoints (21*3)
INPUT_DIM = 63
CONF_THRESHOLD = 20.0
TOP_K = 5
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

def infer_gru_params_from_state_dict(state_dict: dict):
	# Expect keys like 'gru.weight_ih_l0', 'gru.weight_hh_l0', optional l1, etc.
	w_ih_l0 = state_dict.get("gru.weight_ih_l0")
	w_hh_l0 = state_dict.get("gru.weight_hh_l0")
	if w_ih_l0 is None or w_hh_l0 is None:
		raise RuntimeError("Invalid checkpoint: missing GRU layer 0 weights")

	input_size = w_ih_l0.shape[1]
	hidden_size = w_hh_l0.shape[1]

	# Count layers by checking for sequential l{n}
	num_layers = 1
	while f"gru.weight_ih_l{num_layers}" in state_dict:
		num_layers += 1

	# Directionality: if reverse weights exist, it's bidirectional (we don't expect for this ckpt)
	bidirectional = any(k.startswith("gru.weight_ih_l0_reverse") for k in state_dict.keys())
	return input_size, hidden_size, num_layers, bidirectional


def load_labels(gru_dir: str):
	txt_path = os.path.join(gru_dir, "classes_only_1hand.txt")
	csv_path = os.path.join(gru_dir, "word_level_classes.csv")

	labels = None
	if os.path.isfile(txt_path):
		with open(txt_path, "r", encoding="utf-8") as f:
			labels = [line.strip() for line in f if line.strip()]
	elif os.path.isfile(csv_path):
		# Fallback: first column or header names
		try:
			import csv
			with open(csv_path, "r", encoding="utf-8") as f:
				reader = csv.reader(f)
				labels = []
				for row in reader:
					if row:
						labels.append(row[0].strip())
				# Remove header if present
				if labels and labels[0].lower() in ("class", "label", "word"):
					labels = labels[1:]
		except Exception:
			pass
	return labels


def load_checkpoint_and_model():
	base_dir = os.path.dirname(__file__)
	candidates = [
		os.path.join(base_dir, "best_gru_wlasl1hand.pt"),
		os.path.join(base_dir, "word_level_gru.pth"),
	]

	ckpt = None
	ckpt_path = None
	for p in candidates:
		if os.path.isfile(p):
			ckpt_path = p
			break
	if ckpt_path is None:
		raise FileNotFoundError("❌ No GRU checkpoint found (best_gru_wlasl1hand.pt / word_level_gru.pth)")

	ckpt = torch.load(ckpt_path, map_location=DEVICE)

	# Labels may be embedded; otherwise load from files
	labels = ckpt.get("labels")
	if labels is None:
		labels = load_labels(base_dir)
	if labels is None or len(labels) == 0:
		raise RuntimeError("❌ Could not determine label list for GRU model")

	state_dict = None
	if isinstance(ckpt, dict):
		# Try common keys
		state_dict = ckpt.get("state_dict") or ckpt.get("model") or ckpt.get("model_state_dict")
	# If still none and ckpt looks like a state dict
	if state_dict is None and isinstance(ckpt, dict):
		# Heuristic: dict with tensor params
		keys = list(ckpt.keys())
		if keys and isinstance(ckpt[keys[0]], torch.Tensor):
			state_dict = ckpt
	if state_dict is None:
		raise RuntimeError("❌ Checkpoint format not recognized: missing state_dict/model")

	# Infer model params from state dict
	in_size, hid_size, num_layers, bidir = infer_gru_params_from_state_dict(state_dict)
	if bidir:
		raise RuntimeError("❌ This test script expects a unidirectional GRU checkpoint.")

	# Build GRUClassifier architecture used in training
	# Determine number of classes from state_dict head if available
	out_classes = None
	if "fc.3.weight" in state_dict:
		out_classes = state_dict["fc.3.weight"].shape[0]
	elif "fc.weight" in state_dict:
		out_classes = state_dict["fc.weight"].shape[0]
	if out_classes is not None and out_classes != len(labels):
		# Prefer the number defined by the checkpoint and trim labels if longer
		if len(labels) > out_classes:
			labels = labels[:out_classes]
		else:
			raise RuntimeError(f"❌ Label count ({len(labels)}) != checkpoint classes ({out_classes})")

	model = GRUClassifier(
		input_size=in_size,
		hidden_size=hid_size,
		num_layers=num_layers,
		num_classes=len(labels),
		dropout=0.3,
	)

	model.load_state_dict(state_dict, strict=True)
	model.eval()
	model.to(DEVICE)
	return model, labels


model, labels = load_checkpoint_and_model()

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
# Keypoint extraction (63 dims, one-hand)
# ----------------------------
def extract_keypoints_63(res):
	# Prefer right hand; fallback to left; else zeros
	hand = res.right_hand_landmarks or res.left_hand_landmarks
	if hand:
		arr = []
		for lm in hand.landmark:
			arr.extend([lm.x, lm.y, lm.z])
		return np.array(arr, dtype=np.float32)
	return np.zeros(63, dtype=np.float32)


# ----------------------------
# Webcam loop (no-hand reject)
# ----------------------------
sequence = []
NO_HAND_MAX = 5
no_hand_count = 0
while True:
	ret, frame = cap.read()
	if not ret:
		break

	img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	res = holistic.process(img)

	kp = extract_keypoints_63(res)
	# Presence check: any non-zero indicates a detected hand
	if float(np.sum(np.abs(kp))) == 0.0:
		no_hand_count += 1
		if no_hand_count >= NO_HAND_MAX:
			sequence.clear()
			cv2.putText(
				frame, "No sign",
				(20, 40), cv2.FONT_HERSHEY_SIMPLEX,
				1, (0, 0, 255), 2
			)
		cv2.imshow("ASL Recognition (GRU)", frame)
		if cv2.waitKey(1) & 0xFF == 27:
			break
		continue
	else:
		no_hand_count = 0

	sequence.append(kp)
	if len(sequence) > MAX_LEN:
		sequence = sequence[-MAX_LEN:]

	if len(sequence) == MAX_LEN:
		x = torch.from_numpy(np.array(sequence, dtype=np.float32)).unsqueeze(0).to(DEVICE)
		with torch.no_grad():
			out = model(x)
			probs = F.softmax(out, dim=1)
			conf, idx = torch.max(probs, dim=1)
			topk = min(TOP_K, probs.shape[1])
			topk_conf, topk_idx = torch.topk(probs, k=topk, dim=1)
			# Console stats for top-k
			pairs = [f"{labels[i]}: {p.item()*100:.1f}%" for i, p in zip(topk_idx[0].tolist(), topk_conf[0])]
			print("Top-{} -> ".format(topk) + " | ".join(pairs))

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

	cv2.imshow("ASL Recognition (GRU)", frame)
	if cv2.waitKey(1) & 0xFF == 27:
		break

cap.release()
cv2.destroyAllWindows()
