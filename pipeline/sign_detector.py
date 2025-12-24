import torch
import numpy as np
import sys
from pathlib import Path
from collections import deque

# Ensure repo root and module dir on sys.path so imports work anywhere
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

SLD_DIR = ROOT / "sign_language_detection_module"
if str(SLD_DIR) not in sys.path:
    sys.path.append(str(SLD_DIR))

from sign_language_detection_module.train import LSTMClassifier, TransformerClassifier
from sign_language_detection_module.config.config import load_config


class SignLanguageDetector:
    """
    Real-time Sign Language Detector for binary classification (YES/NO).
    Includes temporal smoothing and state management to reduce flickering.
    """
    
    def __init__(self, checkpoint_path, config_path=None, device=None, 
                 smoothing_window=5, hysteresis_upper=0.75, hysteresis_lower=0.65,
                 smoothing_alpha_up=0.6, smoothing_alpha_down=0.3):
        """
        Initialize the sign language detector.
        
        Args:
            checkpoint_path: Path to model checkpoint file
            config_path: Path to config YAML file (optional, will try default)
            device: Device to run inference on ('cuda' or 'cpu', None for auto)
            smoothing_window: Number of predictions to average for temporal smoothing (legacy, kept for compatibility)
            hysteresis_upper: Upper threshold for switching to YES state
            hysteresis_lower: Lower threshold for switching back to NO state
            smoothing_alpha_up: EMA alpha used when confidence is rising (0-1, higher = faster rise)
            smoothing_alpha_down: EMA alpha used when confidence is falling (0-1, lower = slower decay)
        """
        # Load config
        if config_path is None:
            config_path = Path(__file__).parent.parent / "sign_language_detection_module" / "config" / "config.yaml"
        
        self.config = load_config(str(config_path))
        
        # Set device
        if device is None:
            self.device = self.config.get_device()
        else:
            self.device = torch.device(device)
        
        # Initialize model based on config
        if self.config.model.model_type == 'lstm':
            self.model = LSTMClassifier(
                input_dim=126,
                hidden_dim=self.config.model.hidden_dim,
                num_layers=self.config.model.num_layers,
                num_classes=2,
                dropout=self.config.model.dropout
            ).to(self.device)
        elif self.config.model.model_type == 'transformer':
            from sign_language_detection_module.train import TransformerClassifier
            self.model = TransformerClassifier(
                input_dim=126,
                d_model=self.config.model.hidden_dim,
                nhead=8,
                num_layers=self.config.model.num_layers,
                num_classes=2,
                dropout=self.config.model.dropout
            ).to(self.device)
        else:
            raise ValueError(f"Unknown model type: {self.config.model.model_type}")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Temporal smoothing and state management
        self.smoothing_window = smoothing_window
        # EMA with separate rise/decay alphas for quicker responsiveness
        self.smoothing_alpha_up = float(smoothing_alpha_up)
        self.smoothing_alpha_down = float(smoothing_alpha_down)
        self.smoothed_yes = None
        self.hysteresis_upper = hysteresis_upper
        self.hysteresis_lower = hysteresis_lower
        self.current_state = 0  # 0 = NO, 1 = YES
        
        print(f"Loaded Sign Language Detector:")
        print(f"  Model: {self.config.model.model_type}")
        print(f"  Checkpoint: {checkpoint_path}")
        print(f"  Device: {self.device}")
        print(f"  Epoch: {checkpoint.get('epoch', 'unknown') + 1}")
        print(f"  Smoothing (EMA): up={self.smoothing_alpha_up:.2f}, down={self.smoothing_alpha_down:.2f}")
        print(f"  Hysteresis: {hysteresis_lower:.2f} (low) / {hysteresis_upper:.2f} (high)")
    
    def predict(self, sequence, use_smoothing=True):
        """
        Predict if sequence contains sign language.
        
        Args:
            sequence: numpy array of shape (seq_len, 126) or (seq_len, 227)
                - 126: hand-only landmarks (right + left hands)
                - 227: holistic landmarks (pose + hands + presence)
            use_smoothing: Whether to apply temporal smoothing (default: True)
            
        Returns:
            dict with keys:
                - prediction: int (0=NO, 1=YES) - raw model prediction
                - confidence: float (confidence score for predicted class)
                - yes_confidence: float (raw YES probability)
                - no_confidence: float (raw NO probability)
                - smoothed_yes_confidence: float (smoothed YES probability)
                - smoothed_no_confidence: float (smoothed NO probability)
                - stable_state: int (0=NO, 1=YES) - state with hysteresis applied
                - probabilities: dict with 'NO' and 'YES' probabilities
        """
        # Convert to tensor
        if isinstance(sequence, np.ndarray):
            sequence = torch.FloatTensor(sequence)
        
        # Extract hand landmarks if holistic format (227 dims)
        # Holistic format: pose(99) + left_hand(63) + right_hand(63) + presence(2)
        # We need: left_hand(63) + right_hand(63) = indices [99:225]
        # But sign detector expects: right_hand(63) + left_hand(63)
        # So we need to swap: right[99:162] + left[162:225]
        if sequence.shape[-1] == 227:
            # Extract hands and reorder to match training format
            right_hand = sequence[:, 99:162]  # indices 99-161
            left_hand = sequence[:, 162:225]   # indices 162-224
            sequence = torch.cat([right_hand, left_hand], dim=-1)  # (seq_len, 126)
        
        sequence = sequence.unsqueeze(0).to(self.device)  # Add batch dimension
        
        with torch.no_grad():
            outputs = self.model(sequence)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, prediction = torch.max(probabilities, dim=1)
            
            # Get YES class probability (class 1)
            yes_prob = probabilities[0][1].item()
            no_prob = probabilities[0][0].item()
            
            # Apply EMA smoothing for responsiveness
            if use_smoothing:
                if self.smoothed_yes is None:
                    self.smoothed_yes = yes_prob
                else:
                    # Use faster rise and slower decay to reduce perceived lag
                    if yes_prob >= self.smoothed_yes:
                        alpha = self.smoothing_alpha_up
                    else:
                        alpha = self.smoothing_alpha_down
                    self.smoothed_yes = alpha * yes_prob + (1.0 - alpha) * self.smoothed_yes
                smoothed_yes = float(self.smoothed_yes)
                smoothed_no = 1.0 - smoothed_yes
                
                # Apply hysteresis for state management
                if self.current_state == 0:  # Currently NO
                    # Need higher confidence to switch to YES
                    if smoothed_yes >= self.hysteresis_upper:
                        self.current_state = 1
                else:  # Currently YES
                    # Need to drop below lower threshold to switch to NO
                    if smoothed_yes < self.hysteresis_lower:
                        self.current_state = 0
            else:
                smoothed_yes = yes_prob
                smoothed_no = no_prob
            
            return {
                'prediction': prediction.item(),
                'confidence': confidence.item(),
                'yes_confidence': yes_prob,
                'no_confidence': no_prob,
                'smoothed_yes_confidence': smoothed_yes,
                'smoothed_no_confidence': smoothed_no,
                'stable_state': self.current_state,
                'probabilities': {
                    'NO': no_prob,
                    'YES': yes_prob
                }
            }
    
    def is_sign_language(self, sequence, threshold=0.7):
        """
        Check if sequence is sign language based on threshold.
        
        Args:
            sequence: numpy array of shape (seq_len, 126) or (seq_len, 227)
            threshold: Confidence threshold for YES class (default: 0.7)
            
        Returns:
            bool: True if sign language detected above threshold
        """
        result = self.predict(sequence)
        return result['yes_confidence'] >= threshold
    
    def reset_smoothing(self):
        """
        Reset the temporal smoothing buffer and state.
        Useful when starting a new detection session.
        """
        self.smoothed_yes = None
        self.current_state = 0
        print("Smoothing buffer and state reset")

