import cv2
import numpy as np
import mediapipe as mp
import os


class VideoToNpyConverter:
    def __init__(self, seq_len=37, max_hands=2):
        self.seq_len = seq_len
        self.max_hands = max_hands
        self.landmark_dim = 21 * 3 * max_hands

        mp_hands = mp.solutions.hands
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )

    def extract_landmarks(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)

        if not results.multi_hand_landmarks:
            return np.zeros(self.landmark_dim)

        coords = []

        for hand in results.multi_hand_landmarks[:self.max_hands]:
            for lm in hand.landmark:
                coords.extend([lm.x, lm.y, lm.z])

        if len(results.multi_hand_landmarks) < self.max_hands:
            coords.extend([0.0] * (21 * 3))

        return np.array(coords)

    def normalize_sequence(self, seq):
        if len(seq) == 0:
            return np.zeros((self.seq_len, self.landmark_dim))

        indices = np.linspace(0, len(seq) - 1, self.seq_len).astype(int)
        return np.array([seq[i] for i in indices])

    def video_to_npy(self, video_path):
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        sequence = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            lm_vec = self.extract_landmarks(frame)
            sequence.append(lm_vec)

        cap.release()

        sequence = self.normalize_sequence(sequence)

        return sequence

    def save(self, video_path, output_dir):
        os.makedirs(output_dir, exist_ok=True)

        seq = self.video_to_npy(video_path)

        base = os.path.splitext(os.path.basename(video_path))[0]
        out_path = os.path.join(output_dir, base + ".npy")

        np.save(out_path, seq)
        print(f"Saved: {out_path}")

        return out_path


class FramesToNpyConverter:
    def __init__(self, seq_len=37, max_hands=2):
        self.seq_len = seq_len
        self.max_hands = max_hands
        self.landmark_dim = 21 * 3 * max_hands

        mp_hands = mp.solutions.hands
        self.hands = mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=max_hands,
            min_detection_confidence=0.6
        )

    def extract_landmarks(self, image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image_rgb)

        if not results.multi_hand_landmarks:
            return np.zeros(self.landmark_dim, dtype=np.float32)

        coords = []

        for hand in results.multi_hand_landmarks[: self.max_hands]:
            for lm in hand.landmark:
                coords.extend([lm.x, lm.y, lm.z])

        missing = self.max_hands - len(results.multi_hand_landmarks)
        if missing > 0:
            coords.extend([0.0] * missing * 21 * 3)

        return np.array(coords, dtype=np.float32)

    def frames_folder_to_npy(self, frames_dir):
        frame_files = sorted(
            f for f in os.listdir(frames_dir)
            if f.lower().endswith((".jpg", ".png"))
        )

        sequence = []

        for f in frame_files:
            img = cv2.imread(os.path.join(frames_dir, f))
            if img is None:
                continue

            sequence.append(self.extract_landmarks(img))

        if len(sequence) == 0:
            return np.zeros((self.seq_len, self.landmark_dim), dtype=np.float32)

        if len(sequence) != self.seq_len:
            idx = np.linspace(0, len(sequence) - 1, self.seq_len).astype(int)
            sequence = [sequence[i] for i in idx]

        return np.stack(sequence)

    def save(self, frames_dir, output_dir):
        os.makedirs(output_dir, exist_ok=True)

        seq = self.frames_folder_to_npy(frames_dir)

        video_name = os.path.basename(frames_dir.rstrip("/"))
        out_path = os.path.join(output_dir, video_name + ".npy")

        np.save(out_path, seq)
        return out_path
