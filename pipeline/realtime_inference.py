import cv2
import numpy as np
import argparse
from pathlib import Path
import sys
import mediapipe as mp
import torch
import time

# Ensure repo root on sys.path so absolute imports work anywhere
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from pipeline.frame_buffer import FrameBuffer
from pipeline.sign_detector import SignLanguageDetector


class ASLModel(torch.nn.Module):
    """LSTM model for ASL recognition (full body + hands)"""
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.lstm = torch.nn.LSTM(
            input_dim, 256, 2,
            batch_first=True, bidirectional=True
        )
        self.fc = torch.nn.Linear(512, num_classes)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        h = torch.cat([h[-2], h[-1]], dim=1)
        return self.fc(h)


class WordLevelDetector:
    """
    Detects and classifies sign language words using either GRU or LSTM model.
    Supports two model types:
    - GRU: Right hand only (63 dims), 32 frames
    - LSTM: Full body + hands (227 dims), 120 frames
    """
    
    def __init__(self, model_path, classes_path=None, model_type='gru', device=None):
        """
        Initialize the word-level detector.
        
        Args:
            model_path: Path to the model checkpoint (.pt or .pth)
            classes_path: Path to the class names file (optional for LSTM)
            model_type: Model type ('gru' or 'lstm')
            device: Device to run on ('cuda' or 'cpu', None for auto)
        """
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        self.model_type = model_type.lower()
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location='cpu')
        
        if self.model_type == 'gru':
            self._init_gru_model(checkpoint, classes_path)
        elif self.model_type == 'lstm':
            self._init_lstm_model(checkpoint, classes_path)
        else:
            raise ValueError(f"Unknown model type: {model_type}. Use 'gru' or 'lstm'")
        
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Loaded Word-Level Detector ({self.model_type.upper()}):")
        print(f"  Model: {model_path}")
        print(f"  Classes: {len(self.classes)}")
        print(f"  Input dims: {self.input_size}")
        if self.model_type == 'gru':
            print(f"  GRU hands: {'both (126)' if getattr(self, 'uses_both_hands', False) else 'single (63)'}")
        print(f"  Target sequence length: {self.target_len}")
        print(f"  Device: {self.device}")
    
    def _init_gru_model(self, checkpoint, classes_path):
        """Initialize GRU model (right hand only)"""
        # Import GRU model
        sys.path.insert(0, str(ROOT / "word_level_detection_module" / "GRU"))
        from model import GRUClassifier
        
        # Inspect checkpoint structure
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # Infer model dimensions from checkpoint weights
        gru_weight_ih = state_dict.get('gru.weight_ih_l0')
        if gru_weight_ih is not None:
            input_size = gru_weight_ih.shape[1]
            hidden_size = gru_weight_ih.shape[0] // 3
        else:
            input_size = 63
            hidden_size = 128
        
        # Count GRU layers
        num_layers = 1
        while f'gru.weight_ih_l{num_layers}' in state_dict:
            num_layers += 1
        
        # Get number of classes
        fc_out_weight = state_dict.get('fc.3.weight')
        if fc_out_weight is not None:
            num_classes = fc_out_weight.shape[0]
        else:
            num_classes = 27
        
        # Create model
        self.model = GRUClassifier(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_classes=num_classes
        )
        
        # Load weights
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        # Load class names
        if classes_path:
            with open(classes_path, 'r') as f:
                self.classes = [line.strip() for line in f if line.strip()]
        else:
            self.classes = [f"class_{i}" for i in range(num_classes)]
        
        self.input_size = input_size
        self.target_len = 32
        # Use both hands if the GRU was trained for 126-dim input
        self.uses_both_hands = (self.input_size >= 120)
    
    def _init_lstm_model(self, checkpoint, classes_path):
        """Initialize LSTM model (full body + hands)"""
        # LSTM checkpoint format: {'model': state_dict, 'labels': list}
        if isinstance(checkpoint, dict) and 'labels' in checkpoint:
            self.classes = checkpoint['labels']
            state_dict = checkpoint['model']
        else:
            raise ValueError("LSTM checkpoint must contain 'labels' and 'model' keys")
        
        # Create model
        input_dim = 227  # pose(33*3) + left_hand(21*3) + right_hand(21*3) + presence(2)
        num_classes = len(self.classes)
        
        self.model = ASLModel(input_dim, num_classes)
        self.model.load_state_dict(state_dict)
        
        self.input_size = input_dim
        self.target_len = 20  # Match test_ASL.py
    
    def extract_landmarks_for_model(self, full_landmarks):
        """
        Extract appropriate landmarks based on model type.
        
        Args:
            full_landmarks: numpy array from buffer
                - Hands mode: (seq_len, 126) with right_hand(63) + left_hand(63)
                - Holistic mode: (seq_len, 227) with pose(99) + left_hand(63) + right_hand(63) + presence(2)
            
        Returns:
            numpy array with model-specific landmarks
            - GRU: (seq_len, 63 or 126) based on model training
            - LSTM: (seq_len, 227) already in correct format from holistic buffer
        """
        if self.model_type == 'gru':
            # Extract hand landmarks from holistic if needed
            # Holistic format: pose(99) + left_hand(63) + right_hand(63) + presence(2)
            # Target format: right_hand(63) + left_hand(63) = 126
            if full_landmarks.shape[1] == 227:
                right_hand = full_landmarks[:, 99:162]   # indices 99-161
                left_hand = full_landmarks[:, 162:225]    # indices 162-224
                full_landmarks = np.concatenate([right_hand, left_hand], axis=1)  # (seq_len, 126)
            
            # GRU: prefer both hands if the model expects 126-dim input
            if getattr(self, 'uses_both_hands', False) and full_landmarks.shape[1] >= 126:
                return full_landmarks[:, :126].astype(np.float32)

            # Otherwise fall back to single-hand input (choose available hand)
            right = full_landmarks[:, :63]
            left = full_landmarks[:, 63:126] if full_landmarks.shape[1] >= 126 else None

            right_present = np.any(np.abs(right) > 0.01)
            left_present = np.any(np.abs(left) > 0.01) if left is not None else False

            if right_present:
                return right.astype(np.float32)
            if left_present:
                return left.astype(np.float32)
            # Neither hand present; return right (zeros) to trigger NO_HAND downstream
            return right.astype(np.float32)
        
        elif self.model_type == 'lstm':
            # LSTM: landmarks already extracted by holistic mode in buffer (227 dims)
            # pose(99) + left_hand(63) + right_hand(63) + presence(2)
            return full_landmarks.astype(np.float32)
        
        return full_landmarks
    
    def extract_right_hand_landmarks(self, full_landmarks):
        """
        Legacy method - extract right hand landmarks (first 63 dims).
        Kept for backward compatibility.
        """
        return full_landmarks[:, :63].astype(np.float32)
    
    def resize_sequence(self, seq, target_len):
        """
        Resize sequence to target length using linear interpolation.
        
        Args:
            seq: numpy array of shape (T, 63)
            target_len: Target sequence length
            
        Returns:
            numpy array of shape (target_len, 63)
        """
        T, D = seq.shape
        if T == target_len:
            return seq
        
        idx = np.linspace(0, T - 1, target_len)
        resized = np.stack([
            np.interp(idx, np.arange(T), seq[:, d])
            for d in range(D)
        ], axis=1)
        
        return resized.astype(np.float32)
    
    def predict(self, landmarks_sequence, return_top_k=3):
        """
        Predict sign class from hand landmarks sequence.
        
        Args:
            landmarks_sequence: numpy array of shape (seq_len, 126) with full landmarks
            return_top_k: Return top-k predictions (default: 3)
            
        Returns:
            dict with keys:
                - top_classes: List of top-k class names
                - top_confidences: List of top-k confidence scores
                - top_indices: List of top-k class indices
                - predicted_class: Top-1 predicted class name
                - confidence: Top-1 confidence score
        """
        # Extract model-specific landmarks
        model_landmarks = self.extract_landmarks_for_model(landmarks_sequence)
        
        # Check if sequence is mostly zeros (no hand detected)
        if np.allclose(model_landmarks, 0):
            return {
                'predicted_class': 'NO_HAND',
                'confidence': 1.0,
                'top_classes': ['NO_HAND'],
                'top_confidences': [1.0],
                'top_indices': [-1]
            }
        
        # Resize to target length
        resized = self.resize_sequence(model_landmarks, self.target_len)
        
        # Convert to tensor and add batch dimension
        seq_tensor = torch.FloatTensor(resized).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            logits = self.model(seq_tensor)
            probabilities = torch.softmax(logits, dim=1)
            
            # Get top-k predictions
            top_probs, top_indices = torch.topk(probabilities[0], min(return_top_k, len(self.classes)))
            top_probs = top_probs.cpu().numpy()
            top_indices = top_indices.cpu().numpy()
            
            top_classes = [self.classes[idx] if idx < len(self.classes) else f"class_{idx}" 
                          for idx in top_indices]
            
            return {
                'predicted_class': top_classes[0],
                'confidence': float(top_probs[0]),
                'top_classes': top_classes,
                'top_confidences': [float(p) for p in top_probs],
                'top_indices': list(top_indices)
            }


def draw_landmarks(frame, hand_landmarks):
    """
    Draw hand landmarks on frame for visualization.
    Supports both Hands and Holistic results.
    
    Args:
        frame: BGR image frame
        hand_landmarks: MediaPipe hand/holistic landmarks results
    """
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    
    # Handle both hand-only and holistic results
    if hasattr(hand_landmarks, 'multi_hand_landmarks') and hand_landmarks.multi_hand_landmarks:
        # Hands mode
        for hand_landmarks_set in hand_landmarks.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks_set,
                mp.solutions.hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )
    elif hasattr(hand_landmarks, 'left_hand_landmarks') or hasattr(hand_landmarks, 'right_hand_landmarks'):
        # Holistic mode
        if hand_landmarks.right_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks.right_hand_landmarks,
                mp.solutions.holistic.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )
        if hand_landmarks.left_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks.left_hand_landmarks,
                mp.solutions.holistic.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )


def draw_hand_annotation(frame, hand_landmarks, label_text=None):
    """
    Draw bounding boxes around all detected hands with an optional label on the first.
    The label (e.g., "YES: 99.0%") is rendered on the first box to reduce clutter.
    Supports both Hands and Holistic results.
    """
    # Handle both hand-only and holistic results
    hands_to_draw = []
    if hasattr(hand_landmarks, 'multi_hand_landmarks') and hand_landmarks.multi_hand_landmarks:
        hands_to_draw = hand_landmarks.multi_hand_landmarks
    elif hasattr(hand_landmarks, 'left_hand_landmarks') or hasattr(hand_landmarks, 'right_hand_landmarks'):
        # Holistic mode
        if hand_landmarks.right_hand_landmarks:
            hands_to_draw.append(hand_landmarks.right_hand_landmarks)
        if hand_landmarks.left_hand_landmarks:
            hands_to_draw.append(hand_landmarks.left_hand_landmarks)
    
    if not hands_to_draw:
        return
    
    h, w = frame.shape[:2]
    color = (0, 255, 0)
    thickness = 2

    for idx, hand in enumerate(hands_to_draw):
        xs = [int(lm.x * w) for lm in hand.landmark]
        ys = [int(lm.y * h) for lm in hand.landmark]
        x_min, x_max = max(min(xs), 0), min(max(xs), w - 1)
        y_min, y_max = max(min(ys), 0), min(max(ys), h - 1)
        pad = 10
        x_min = max(x_min - pad, 0)
        y_min = max(y_min - pad, 0)
        x_max = min(x_max + pad, w - 1)
        y_max = min(y_max + pad, h - 1)
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, thickness)

        # Only draw the text label on the first hand
        if idx == 0 and label_text:
            (tw, th), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            box_y = max(y_min - th - 6, 0)
            cv2.rectangle(frame, (x_min, box_y), (x_min + tw + 8, box_y + th + 6), color, -1)
            cv2.putText(frame, label_text, (x_min + 4, box_y + th + 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)


def draw_info(frame, buffer_status):
    """Minimal on-screen info; detailed metrics are logged to console."""
    cv2.putText(frame, f"Buffer: {buffer_status}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


def save_clip(frames, out_dir, fps, filename_stub):
    """
    Save a sequence of BGR frames to an MP4 file.

    Args:
        frames: list of numpy arrays (H, W, 3) in BGR
        out_dir: Path-like output directory
        fps: frames per second for the writer
        filename_stub: base name components (without extension)

    Returns:
        Path to the saved clip, or None if nothing was saved
    """
    if not frames:
        return None

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    h, w = frames[0].shape[:2]
    # Try H264, fall back to mp4v if unavailable
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    ts = time.strftime('%Y%m%d_%H%M%S')
    out_path = out_dir / f"{filename_stub}_{ts}.mp4"
    writer = cv2.VideoWriter(str(out_path), fourcc, max(fps, 1), (w, h))

    for f in frames:
        if f.shape[:2] != (h, w):
            f = cv2.resize(f, (w, h))
        writer.write(f)

    writer.release()
    return out_path


def main():
    parser = argparse.ArgumentParser(description='Real-time Sign Language Detection with Word Classification')
    parser.add_argument('--checkpoint', type=str,
                       default='sign_language_detection_module/checkpoints/val_acc_main_metric/sl_binary_classifier_epoch35.pth',
                       help='Path to sign detection model checkpoint')
    
    # Word classification model options
    parser.add_argument('--word-model-type', type=str, choices=['gru', 'lstm'], default='lstm',
                       help='Word classification model type: gru or lstm (default: gru)')
    parser.add_argument('--word-model', type=str, default=None,
                       help='Path to word-level model (auto-detected if not specified)')
    parser.add_argument('--word-classes', type=str, default=None,
                       help='Path to word classes file (auto-detected if not specified)')
    
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config YAML file (optional)')
    parser.add_argument('--camera', type=int, default=0,
                       help='Camera device index (default: 0)')
    parser.add_argument('--threshold', type=float, default=0.7,
                       help='Confidence threshold for YES class (default: 0.5)')
    parser.add_argument('--word-threshold', type=float, default=0.3,
                       help='Confidence threshold for word classification (default: 0.3)')
    parser.add_argument('--skip-frames', type=int, default=1,
                       help='Run inference every N frames (default: 1)')
    parser.add_argument('--mode', type=str, choices=['sign-only', 'word-only', 'combined'], default='word-only',
                       help='Pipeline mode: sign-only (binary YES/NO detection), word-only (word classification), or combined (sign detector gates word classification)')
    parser.add_argument('--max-hands', type=int, default=2,
                       help='Maximum number of hands to detect (default: 2)')
    parser.add_argument('--smoothing-window', type=int, default=5,
                       help='Number of predictions to smooth over (legacy, unused with EMA)')
    parser.add_argument('--hysteresis-upper', type=float, default=0.70,
                       help='Upper threshold for switching to YES (default: 0.80)')
    parser.add_argument('--hysteresis-lower', type=float, default=0.60,
                       help='Lower threshold for switching to NO (default: 0.70)')
    parser.add_argument('--ema-up', type=float, default=0.6,
                       help='EMA alpha when confidence rises (higher = faster rise)')
    parser.add_argument('--ema-down', type=float, default=0.3,
                       help='EMA alpha when confidence falls (lower = slower decay)')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda/cpu, default: auto)')
    parser.add_argument('--hand-detection-confidence', type=float, default=0.5,
                       help='MediaPipe hand detection confidence (default: 0.5, lower = more sensitive)')
    parser.add_argument('--hand-tracking-confidence', type=float, default=0.5,
                       help='MediaPipe hand tracking confidence (default: 0.5, lower = more sensitive)')
    parser.add_argument('--save-clips', action='store_true',
                       help='Save the current buffered frames as MP4 when a word prediction occurs')
    parser.add_argument('--clip-dir', type=str, default='results/clips',
                       help='Directory to store saved clips (default: results/clips)')
    
    args = parser.parse_args()
    
    # Auto-detect word model and classes paths based on model type
    if args.word_model is None:
        if args.word_model_type == 'gru':
            args.word_model = 'word_level_detection_module/GRU/best_gru_wlasl1hand.pt'
        else:  # lstm
            args.word_model = 'word_level_detection_module/LSTM/asl_lstm_keypoint_model.pth'
    
    if args.word_classes is None and args.word_model_type == 'gru':
        args.word_classes = 'word_level_detection_module/GRU/classes_only_1hand.txt'
    # LSTM classes are embedded in checkpoint, no separate file needed
    
    # Resolve checkpoint paths
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = Path(__file__).parent.parent / checkpoint_path
    
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        sys.exit(1)
    
    word_model_path = Path(args.word_model)
    if not word_model_path.is_absolute():
        word_model_path = Path(__file__).parent.parent / word_model_path
    
    if not word_model_path.exists():
        print(f"Error: Word model not found at {word_model_path}")
        sys.exit(1)
    
    # Word classes path (only for GRU)
    word_classes_path = None
    if args.word_classes:
        word_classes_path = Path(args.word_classes)
        if not word_classes_path.is_absolute():
            word_classes_path = Path(__file__).parent.parent / word_classes_path
        
        if not word_classes_path.exists():
            print(f"Error: Classes file not found at {word_classes_path}")
            sys.exit(1)
    
    print("=" * 80)
    print("Sign Language Detection + Word Classification (Real-time)")
    print("=" * 80)
    print(f"Mode: {args.mode}")
    if args.mode in ['combined', 'sign-only']:
        print(f"Sign Detection Checkpoint: {checkpoint_path}")
    if args.mode in ['combined', 'word-only']:
        print(f"Word Model Type: {args.word_model_type.upper()}")
        print(f"Word Classification Model: {word_model_path}")
        if word_classes_path:
            print(f"Classes: {word_classes_path}")
    print(f"Sign Threshold: {args.threshold}")
    if args.mode in ['combined', 'word-only']:
        print(f"Word Threshold: {args.word_threshold}")
    print(f"Camera: {args.camera}")
    print(f"Inference every {args.skip_frames} frames")
    print(f"EMA: up={args.ema_up:.2f}, down={args.ema_down:.2f}")
    print(f"Hysteresis: {args.hysteresis_lower:.2f} / {args.hysteresis_upper:.2f}")
    print(f"Hand Detection: confidence={args.hand_detection_confidence:.2f}, tracking={args.hand_tracking_confidence:.2f}, max_hands={args.max_hands}")
    print("=" * 80)
    print("\nPress 'q' to quit")
    print("Press 'r' to reset buffer and smoothing")
    if args.mode in ['combined', 'word-only']:
        print("Press 'w' to toggle word classification display")
    print("=" * 80 + "\n")
    
    # Initialize detectors
    sign_detector = None
    if args.mode in ['combined', 'sign-only']:
        sign_detector = SignLanguageDetector(
            checkpoint_path=str(checkpoint_path),
            config_path=args.config,
            device=args.device,
            smoothing_window=args.smoothing_window,
            hysteresis_upper=args.hysteresis_upper,
            hysteresis_lower=args.hysteresis_lower,
            smoothing_alpha_up=args.ema_up,
            smoothing_alpha_down=args.ema_down
        )
    
    word_detector = None
    if args.mode in ['combined', 'word-only']:
        word_detector = WordLevelDetector(
            model_path=str(word_model_path),
            classes_path=str(word_classes_path) if word_classes_path else None,
            model_type=args.word_model_type,
            device=args.device
        )
    
    # Initialize frame buffer with adjustable hand detection sensitivity
    # Use holistic mode if LSTM or if combined/sign-only mode (for sign detector)
    buffer_mode = 'holistic' if (args.word_model_type == 'lstm' or args.mode in ['combined', 'sign-only']) else 'hands'
    # Buffer length: 20 for LSTM, 37 for sign detector/GRU
    if args.mode == 'word-only':
        buffer_seq_len = 20 if args.word_model_type == 'lstm' else 20
    else:
        buffer_seq_len = 20  # sign-only or combined use sign detector's length
    
    frame_buffer = FrameBuffer(
        seq_len=buffer_seq_len, 
        max_hands=args.max_hands,
        min_detection_confidence=args.hand_detection_confidence,
        min_tracking_confidence=args.hand_tracking_confidence,
        mode=buffer_mode
    )
    
    print(f"Frame buffer: mode={buffer_mode}, seq_len={buffer_seq_len}")
    
    # Initialize camera
    cap = cv2.VideoCapture(args.camera)
    
    if not cap.isOpened():
        print(f"Error: Cannot open camera {args.camera}")
        sys.exit(1)
    
    # Set camera properties for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    
    frame_count = 0
    prediction_result = None
    word_result = None
    prev_state = None
    show_word_classification = True
    last_log_time = time.time()
    log_interval = 1.0  # Log every second
    
    if args.mode == 'sign-only':
        window_title = 'Sign Language Detection (YES/NO)'
    elif args.mode == 'word-only':
        window_title = 'Word Classification'
    else:
        window_title = 'Sign Language Detection + Word Classification'

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to read frame from camera")
                break
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Add frame to buffer
            buffer_ready = frame_buffer.add_frame(frame)
            
            # Get hand landmarks
            hand_landmarks = frame_buffer.get_hand_landmarks(frame)
            
            # Run inference as soon as we have a minimal sequence
            if frame_count % args.skip_frames == 0:
                # Allow early inference with padding to reduce latency
                # Require at least 10 frames to avoid very noisy estimates
                if len(frame_buffer.buffer) >= 10:
                    sequence = frame_buffer.get_sequence()
                    
                    if args.mode == 'sign-only':
                        # Sign detection only
                        prediction_result = sign_detector.predict(sequence)
                        stable_state = prediction_result.get('stable_state', prediction_result['prediction'])
                        smoothed_yes = prediction_result.get('smoothed_yes_confidence', prediction_result['yes_confidence'])
                        if prev_state != stable_state:
                            state_str = "YES" if stable_state == 1 else "NO"
                            print(f"State -> {state_str} | smoothed={smoothed_yes:.3f}")
                            prev_state = stable_state
                    
                    elif args.mode == 'combined':
                        # Sign detection gates word classification
                        prediction_result = sign_detector.predict(sequence)
                        stable_state = prediction_result.get('stable_state', prediction_result['prediction'])
                        smoothed_yes = prediction_result.get('smoothed_yes_confidence', prediction_result['yes_confidence'])
                        if stable_state == 1 and smoothed_yes >= args.hysteresis_upper:
                            word_result = word_detector.predict(sequence, return_top_k=3)
                            if prev_state is None or stable_state != prev_state:
                                print(f"State -> YES | smoothed={smoothed_yes:.3f}")
                                prev_state = stable_state
                            if show_word_classification:
                                print(f"  Word: {word_result['predicted_class']} ({word_result['confidence']:.3f})")
                                print(f"    Top-3: {', '.join([f'{c}({conf:.2f})' for c, conf in zip(word_result['top_classes'][:3], word_result['top_confidences'][:3])])}")
                            # Save debugging clip if requested and confident enough
                            if args.save_clips and word_result and word_result.get('confidence', 0.0) >= args.word_threshold:
                                frames = frame_buffer.get_frame_sequence()
                                stub = f"combined_{word_result['predicted_class']}_{word_result['confidence']:.2f}_len{len(frames)}"
                                out_path = save_clip(frames, args.clip_dir, fps, stub)
                                if out_path:
                                    print(f"Saved clip: {out_path} (frames={len(frames)})")
                        else:
                            word_result = None
                            if prev_state != stable_state:
                                print(f"State -> NO | smoothed={smoothed_yes:.3f}")
                                prev_state = stable_state
                    
                    else:  # word-only mode
                        # Word-only mode: always classify words
                        word_result = word_detector.predict(sequence, return_top_k=3)
                        # Save debugging clip if requested and confident enough
                        if args.save_clips and word_result and word_result.get('confidence', 0.0) >= args.word_threshold:
                            frames = frame_buffer.get_frame_sequence()
                            stub = f"wordonly_{word_result['predicted_class']}_{word_result['confidence']:.2f}_len{len(frames)}"
                            out_path = save_clip(frames, args.clip_dir, fps, stub)
                            if out_path:
                                print(f"Saved clip: {out_path} (frames={len(frames)})")
            
            # Per-second console logging
            current_time = time.time()
            if current_time - last_log_time >= log_interval:
                if args.mode == 'sign-only':
                    # Log sign detector state only
                    if prediction_result:
                        stable_state = prediction_result.get('stable_state', prediction_result['prediction'])
                        smoothed_yes = prediction_result.get('smoothed_yes_confidence', prediction_result['yes_confidence'])
                        raw_yes = prediction_result.get('yes_confidence', 0.0)
                        state_str = "YES" if stable_state == 1 else "NO"
                        print(f"[{current_time:.1f}s] SignDetector: State={state_str} | Raw={raw_yes*100:.1f}% | Smoothed={smoothed_yes*100:.1f}% | Thresholds=[{args.hysteresis_lower*100:.0f}%-{args.hysteresis_upper*100:.0f}%] | Buffer={len(frame_buffer.buffer)}/{frame_buffer.seq_len}")
                
                elif args.mode == 'combined':
                    # Log sign detector state
                    if prediction_result:
                        stable_state = prediction_result.get('stable_state', prediction_result['prediction'])
                        smoothed_yes = prediction_result.get('smoothed_yes_confidence', prediction_result['yes_confidence'])
                        raw_yes = prediction_result.get('yes_confidence', 0.0)
                        state_str = "YES" if stable_state == 1 else "NO"
                        print(f"[{current_time:.1f}s] SignDetector: State={state_str} | Raw={raw_yes*100:.1f}% | Smoothed={smoothed_yes*100:.1f}% | Thresholds=[{args.hysteresis_lower*100:.0f}%-{args.hysteresis_upper*100:.0f}%] | Buffer={len(frame_buffer.buffer)}/{frame_buffer.seq_len}")
                    
                    # Log word classification if available
                    if word_result:
                        conf_pct = word_result['confidence'] * 100
                        top3_str = ', '.join([f'{c}({conf*100:.1f}%)' for c, conf in zip(word_result['top_classes'][:3], word_result['top_confidences'][:3])])
                        print(f"[{current_time:.1f}s] Word: {word_result['predicted_class']} ({conf_pct:.1f}%) | Top-3: {top3_str}")
                
                else:  # word-only mode
                    if word_result:
                        conf_pct = word_result['confidence'] * 100
                        top3_str = ', '.join([f'{c}({conf*100:.1f}%)' for c, conf in zip(word_result['top_classes'][:3], word_result['top_confidences'][:3])])
                        print(f"[{current_time:.1f}s] Current: {word_result['predicted_class']} ({conf_pct:.1f}%) | Top-3: {top3_str} | Buffer: {len(frame_buffer.buffer)}/{frame_buffer.seq_len}")
                    else:
                        print(f"[{current_time:.1f}s] No prediction | Buffer: {len(frame_buffer.buffer)}/{frame_buffer.seq_len}")
                last_log_time = current_time
            
            # Display buffer status
            buffer_status = f"{len(frame_buffer.buffer)}/{frame_buffer.seq_len} frames"
            if buffer_ready:
                buffer_status += " (Ready)"
            else:
                buffer_status += " (Buffering...)"
            
            # Minimal info overlay
            draw_info(frame, buffer_status)
            
            # Draw hand landmarks
            draw_landmarks(frame, hand_landmarks)
            
            # Draw bounding box + label tied to gesture
            label = None
            if args.mode == 'sign-only':
                if prediction_result:
                    smoothed_yes = prediction_result.get('smoothed_yes_confidence', prediction_result['yes_confidence'])
                    stable_state = prediction_result.get('stable_state', prediction_result['prediction'])
                    if stable_state == 1:
                        label = f"YES: {smoothed_yes*100:.1f}%"
                    else:
                        label = f"NO: {(1-smoothed_yes)*100:.1f}%"
            elif args.mode == 'combined':
                if prediction_result:
                    smoothed_yes = prediction_result.get('smoothed_yes_confidence', prediction_result['yes_confidence'])
                    stable_state = prediction_result.get('stable_state', prediction_result['prediction'])
                    if stable_state == 1 and smoothed_yes >= args.hysteresis_upper:
                        label = f"YES: {smoothed_yes*100:.1f}%"
                        if word_result and show_word_classification:
                            label += f"\n{word_result['predicted_class']}"
            else:  # word-only
                if word_result and word_result.get('confidence', 0.0) >= args.word_threshold:
                    label = f"{word_result['predicted_class']}: {word_result['confidence']*100:.1f}%"
            draw_hand_annotation(frame, hand_landmarks, label)
            
            # Display frame
            cv2.imshow(window_title, frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                frame_buffer.reset()
                if sign_detector is not None:
                    sign_detector.reset_smoothing()
                prediction_result = None
                word_result = None
                print("Buffer and smoothing reset")
            elif key == ord('w') and args.mode in ['combined', 'word-only']:
                show_word_classification = not show_word_classification
                print(f"Word classification display: {'ON' if show_word_classification else 'OFF'}")
            
            frame_count += 1
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("\nCamera released. Goodbye!")


if __name__ == '__main__':
    main()

