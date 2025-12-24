import numpy as np
import cv2
import mediapipe as mp
from collections import deque


class FrameBuffer:
    """
    Manages a sliding window buffer of hand landmark sequences for real-time inference.
    """
    
    def __init__(self, seq_len=37, max_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5, mode='hands'):
        """
        Initialize frame buffer.
        
        Args:
            seq_len: Number of frames in sequence (default: 37)
            max_hands: Maximum number of hands to detect (default: 2)
            min_detection_confidence: MediaPipe detection confidence (default: 0.5)
            min_tracking_confidence: MediaPipe tracking confidence (default: 0.5)
            mode: 'hands' for hand-only or 'holistic' for pose+hands (default: 'hands')
        """
        self.seq_len = seq_len
        self.max_hands = max_hands
        self.mode = mode
        
        if mode == 'holistic':
            self.landmark_dim = 227  # pose(99) + left_hand(63) + right_hand(63) + presence(2)
            mp_holistic = mp.solutions.holistic
            self.holistic = mp_holistic.Holistic(
                static_image_mode=False,
                model_complexity=1,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence
            )
            self.hands = None
        else:
            self.landmark_dim = 21 * 3 * max_hands  # 126 for 2 hands
            # Initialize MediaPipe hands
            mp_hands = mp.solutions.hands
            self.hands = mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=max_hands,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence
            )
            self.holistic = None
        
        # Buffer to store landmark sequences
        self.buffer = deque(maxlen=seq_len)
        # Parallel buffer to store raw frames aligned with landmarks
        self.frame_buffer = deque(maxlen=seq_len)
        # Track consecutive frames with no hands for reset logic
        self.no_hand_count = 0
        self.no_hand_max = 5
        
    def extract_landmarks(self, frame):
        """
        Extract landmarks from a single frame based on mode.
        
        Args:
            frame: BGR image frame from OpenCV
            
        Returns:
            numpy array of shape (landmark_dim,) with landmarks
        """
        if self.mode == 'holistic':
            return self.extract_keypoints_holistic(frame)
        else:
            return self.extract_keypoints_hands(frame)
    
    def extract_keypoints_hands(self, frame):
        """
        Extract hand landmarks only (126 dims for 2 hands).
        """
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)
        
        if not results.multi_hand_landmarks:
            return np.zeros(self.landmark_dim, dtype=np.float32)
        
        coords = []
        
        # Extract landmarks for up to max_hands
        for hand in results.multi_hand_landmarks[:self.max_hands]:
            for lm in hand.landmark:
                coords.extend([lm.x, lm.y, lm.z])
        
        # Pad with zeros if fewer than max_hands detected
        if len(results.multi_hand_landmarks) < self.max_hands:
            missing_hands = self.max_hands - len(results.multi_hand_landmarks)
            coords.extend([0.0] * (missing_hands * 21 * 3))
        
        return np.array(coords, dtype=np.float32)
    
    def extract_keypoints_holistic(self, frame):
        """
        Extract full pose + hands landmarks (227 dims) for LSTM model.
        
        Returns:
            numpy array: pose(99) + left_hand(63) + right_hand(63) + presence(2)
        """
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.holistic.process(frame_rgb)
        
        keypoints = []
        
        # Pose landmarks (33 x 3 = 99)
        if results.pose_landmarks:
            for lm in results.pose_landmarks.landmark:
                keypoints.extend([lm.x, lm.y, lm.z])
        else:
            keypoints.extend([0.0] * 99)
        
        # Left hand landmarks (21 x 3 = 63)
        if results.left_hand_landmarks:
            for lm in results.left_hand_landmarks.landmark:
                keypoints.extend([lm.x, lm.y, lm.z])
        else:
            keypoints.extend([0.0] * 63)
        
        # Right hand landmarks (21 x 3 = 63)
        if results.right_hand_landmarks:
            for lm in results.right_hand_landmarks.landmark:
                keypoints.extend([lm.x, lm.y, lm.z])
        else:
            keypoints.extend([0.0] * 63)
        
        # Presence flags (2)
        keypoints.extend([
            float(results.left_hand_landmarks is not None),
            float(results.right_hand_landmarks is not None)
        ])
        
        return np.array(keypoints, dtype=np.float32)
    
    def add_frame(self, frame):
        """
        Add a frame to the buffer and extract landmarks.
        Implements no-hand reset logic for holistic mode.
        
        Args:
            frame: BGR image frame from OpenCV
            
        Returns:
            bool: True if buffer is full (ready for inference), False otherwise
        """
        landmarks = self.extract_landmarks(frame)
        
        # Check for no-hand condition in holistic mode
        if self.mode == 'holistic':
            # Last 2 values are presence flags
            lh_present, rh_present = landmarks[-2], landmarks[-1]
            if lh_present == 0 and rh_present == 0:
                self.no_hand_count += 1
                if self.no_hand_count >= self.no_hand_max:
                    self.reset()
                    return False
            else:
                self.no_hand_count = 0
        
        self.buffer.append(landmarks)
        # Store a copy of the frame to avoid accidental mutation downstream
        self.frame_buffer.append(frame.copy())
        
        return len(self.buffer) == self.seq_len
    
    def get_sequence(self):
        """
        Get the current sequence from buffer.
        
        Returns:
            numpy array of shape (seq_len, landmark_dim) ready for model input
        """
        if len(self.buffer) < self.seq_len:
            # Pad with zeros if buffer not full
            padding = [np.zeros(self.landmark_dim, dtype=np.float32)] * (self.seq_len - len(self.buffer))
            sequence = list(self.buffer) + padding
        else:
            sequence = list(self.buffer)
        
        return np.array(sequence, dtype=np.float32)
    
    def is_ready(self):
        """
        Check if buffer has enough frames for inference.
        
        Returns:
            bool: True if buffer is full
        """
        return len(self.buffer) == self.seq_len
    
    def reset(self):
        """Reset the buffer."""
        self.buffer.clear()
        self.frame_buffer.clear()
        self.no_hand_count = 0
    
    def get_hand_landmarks(self, frame):
        """
        Get MediaPipe hand landmarks results for visualization.
        
        Args:
            frame: BGR image frame from OpenCV
            
        Returns:
            MediaPipe hand/holistic landmarks results
        """
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if self.mode == 'holistic':
            results = self.holistic.process(frame_rgb)
        else:
            results = self.hands.process(frame_rgb)
        return results

    def get_frame_sequence(self):
        """
        Get the current raw frame sequence aligned with the landmark buffer.

        Returns:
            list of BGR frames (len == current buffer length). If fewer than
            seq_len frames are present, returns only available frames to avoid
            synthetic padding in saved clips.
        """
        return list(self.frame_buffer)

