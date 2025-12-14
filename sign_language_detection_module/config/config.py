from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Optional
import yaml
from pathlib import Path
import torch


class WLASLPreprocessingConfig(BaseModel):
    """WLASL dataset preprocessing configuration."""
    
    frames_per_second: int = Field(
        default=5,
        description="Number of frames to extract per second of video",
        gt=0
    )
    max_frames_per_video: int = Field(
        default=37,
        description="Maximum number of frames to keep per video",
        gt=0
    )
    trim_start_portion: float = Field(
        default=0.15,
        description="Trim video start portion (0.0 to 1.0)",
        ge=0.0,
        le=1.0
    )
    trim_end_portion: float = Field(
        default=0.85,
        description="Trim video end portion (0.0 to 1.0)",
        ge=0.0,
        le=1.0
    )
    
    @field_validator('trim_end_portion')
    @classmethod
    def validate_trim_portions(cls, v, info):
        """Ensure trim_end_portion > trim_start_portion."""
        if 'trim_start_portion' in info.data:
            if v <= info.data['trim_start_portion']:
                raise ValueError("trim_end_portion must be greater than trim_start_portion")
        return v


class ModelConfig(BaseModel):
    """Model architecture configuration."""
    
    model_type: str = Field(
        default="lstm",
        description="Model architecture type: lstm, transformer"
    )
    num_classes: int = Field(
        default=2,
        description="Number of classes (binary classification)",
        gt=0
    )
    hidden_dim: int = Field(
        default=256,
        description="Hidden dimension size",
        gt=0
    )
    num_layers: int = Field(
        default=2,
        description="Number of model layers",
        gt=0
    )
    dropout: float = Field(
        default=0.3,
        description="Dropout rate",
        ge=0.0,
        le=1.0
    )
    
    @field_validator('model_type')
    @classmethod
    def validate_model_type(cls, v):
        allowed = ["lstm", "transformer"]
        if v.lower() not in allowed:
            raise ValueError(f"Model type must be one of {allowed}, got {v}")
        return v.lower()


class DataConfig(BaseModel):
    """Dataset configuration."""
    
    dataset_name: str = Field(
        default="sign_language_binary",
        description="Dataset name"
    )
    data_root: str = Field(
        default="./data",
        description="Root directory for datasets"
    )
    train_split: float = Field(
        default=0.8,
        description="Training data split ratio",
        gt=0.0,
        lt=1.0
    )
    val_split: float = Field(
        default=0.1,
        description="Validation data split ratio",
        gt=0.0,
        lt=1.0
    )
    test_split: float = Field(
        default=0.1,
        description="Test data split ratio",
        gt=0.0,
        lt=1.0
    )
    num_workers: int = Field(
        default=4,
        description="Number of workers for data loading",
        ge=0
    )
    pin_memory: bool = Field(
        default=True,
        description="Pin memory for faster GPU transfer"
    )
    
    @field_validator('test_split')
    @classmethod
    def validate_splits(cls, v, info):
        """Ensure train + val + test = 1.0."""
        if 'train_split' in info.data and 'val_split' in info.data:
            total = info.data['train_split'] + info.data['val_split'] + v
            if not (0.99 <= total <= 1.01):  # Allow small floating point errors
                raise ValueError(f"train_split + val_split + test_split must equal 1.0, got {total}")
        return v


class AugmentationConfig(BaseModel):
    """Data augmentation configuration (for landmark data)."""
    
    use_augmentation: bool = Field(
        default=False,
        description="Enable data augmentation for landmarks"
    )
    random_horizontal_flip: bool = Field(
        default=False,
        description="Enable random horizontal flip of landmarks"
    )
    rotation_degrees: int = Field(
        default=0,
        description="Maximum rotation degrees",
        ge=0,
        le=180
    )


class TrainingConfig(BaseModel):
    """Training configuration."""
    
    batch_size: int = Field(
        default=32,
        description="Batch size for training",
        gt=0
    )
    epochs: int = Field(
        default=50,
        description="Number of training epochs",
        gt=0
    )
    learning_rate: float = Field(
        default=0.001,
        description="Initial learning rate",
        gt=0.0
    )
    weight_decay: float = Field(
        default=0.0001,
        description="Weight decay for optimizer",
        ge=0.0
    )
    optimizer: str = Field(
        default="adam",
        description="Optimizer type: adam, sgd, adamw"
    )
    scheduler: str = Field(
        default="cosine",
        description="LR scheduler: cosine, step, plateau, none"
    )
    scheduler_patience: int = Field(
        default=5,
        description="Patience for ReduceLROnPlateau scheduler",
        gt=0
    )
    scheduler_step_size: int = Field(
        default=10,
        description="Step size for StepLR scheduler",
        gt=0
    )
    scheduler_gamma: float = Field(
        default=0.1,
        description="Gamma for StepLR scheduler",
        gt=0.0,
        le=1.0
    )
    gradient_clip: Optional[float] = Field(
        default=1.0,
        description="Gradient clipping value (None to disable)"
    )
    early_stopping_patience: int = Field(
        default=10,
        description="Epochs to wait before early stopping",
        gt=0
    )
    
    @field_validator('optimizer')
    @classmethod
    def validate_optimizer(cls, v):
        allowed = ["adam", "sgd", "adamw"]
        if v.lower() not in allowed:
            raise ValueError(f"Optimizer must be one of {allowed}, got {v}")
        return v.lower()
    
    @field_validator('scheduler')
    @classmethod
    def validate_scheduler(cls, v):
        allowed = ["cosine", "step", "plateau", "none"]
        if v.lower() not in allowed:
            raise ValueError(f"Scheduler must be one of {allowed}, got {v}")
        return v.lower()


class DeviceConfig(BaseModel):
    """Device configuration."""
    
    use_cuda: bool = Field(
        default=True,
        description="Use CUDA if available"
    )
    device_id: int = Field(
        default=0,
        description="CUDA device ID",
        ge=0
    )
    mixed_precision: bool = Field(
        default=False,
        description="Use mixed precision training (AMP)"
    )
    benchmark: bool = Field(
        default=True,
        description="Enable cudnn.benchmark for performance"
    )


class LoggingConfig(BaseModel):
    """Logging and output configuration."""
    
    log_dir: str = Field(
        default="./logs",
        description="TensorBoard log directory"
    )
    checkpoint_dir: str = Field(
        default="./checkpoints",
        description="Directory to save model checkpoints"
    )
    results_dir: str = Field(
        default="./results",
        description="Directory to save results"
    )
    save_best_only: bool = Field(
        default=True,
        description="Only save the best model"
    )
    checkpoint_interval: int = Field(
        default=5,
        description="Save checkpoint every N epochs",
        gt=0
    )


class SLDetectorConfig(BaseModel):
    """Main Sign Language Detector configuration."""
    
    preprocessing: WLASLPreprocessingConfig = Field(
        default_factory=WLASLPreprocessingConfig
    )
    model: ModelConfig = Field(
        default_factory=ModelConfig
    )
    data: DataConfig = Field(
        default_factory=DataConfig
    )
    augmentation: AugmentationConfig = Field(
        default_factory=AugmentationConfig
    )
    training: TrainingConfig = Field(
        default_factory=TrainingConfig
    )
    device: DeviceConfig = Field(
        default_factory=DeviceConfig
    )
    logging: LoggingConfig = Field(
        default_factory=LoggingConfig
    )
    
    # Experiment metadata
    experiment_name: str = Field(
        default="sl_detector_exp",
        description="Name of the experiment"
    )
    seed: int = Field(
        default=42,
        description="Random seed for reproducibility",
        ge=0
    )
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'SLDetectorConfig':
        """
        Load configuration from a YAML file.
        
        Args:
            yaml_path: Path to the YAML configuration file
            
        Returns:
            SLDetectorConfig: Validated configuration object
        """
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {yaml_path}")
        
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return cls(**config_dict)
    
    def to_yaml(self, yaml_path: str) -> None:
        """
        Save configuration to a YAML file.
        
        Args:
            yaml_path: Path to save the YAML configuration file
        """
        yaml_path = Path(yaml_path)
        yaml_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(yaml_path, 'w') as f:
            yaml.dump(
                self.model_dump(),
                f,
                default_flow_style=False,
                sort_keys=False,
                indent=2
            )
    
    def get_device(self) -> torch.device:
        """
        Get the torch device based on configuration.
        
        Returns:
            torch.device: Configured device for training/inference
        """
        if self.device.use_cuda and torch.cuda.is_available():
            return torch.device(f"cuda:{self.device.device_id}")
        return torch.device("cpu")
    
    def format_path(self, path: str, **kwargs) -> str:
        """
        Format path with placeholder replacement.
        
        Args:
            path: Path string with placeholders
            **kwargs: Values to replace placeholders
            
        Returns:
            str: Formatted path string
        """
        return path.format(**kwargs)
    
    def create_directories(self) -> None:
        """Create all necessary directories for the experiment."""
        directories = [
            self.preprocessing.output_img_dir,
            self.logging.log_dir,
            self.logging.checkpoint_dir,
            self.logging.results_dir,
            self.data.data_root
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def get_checkpoint_path(self, epoch: Optional[int] = None, 
                           is_best: bool = False) -> str:
        """
        Get checkpoint path with experiment name.
        
        Args:
            epoch: Epoch number (None for latest)
            is_best: Whether this is the best model
            
        Returns:
            str: Checkpoint file path
        """
        checkpoint_dir = Path(self.logging.checkpoint_dir)
        
        if is_best:
            return str(checkpoint_dir / f"{self.experiment_name}_best.pth")
        elif epoch is not None:
            return str(checkpoint_dir / f"{self.experiment_name}_epoch{epoch}.pth")
        else:
            return str(checkpoint_dir / f"{self.experiment_name}_latest.pth")
    
    def print_config(self) -> None:
        """Print configuration in a readable format."""
        print("=" * 80)
        print(f"Sign Language Detector Configuration: {self.experiment_name}")
        print("=" * 80)
        
        sections = {
            "Preprocessing": self.preprocessing,
            "Model": self.model,
            "Data": self.data,
            "Augmentation": self.augmentation,
            "Training": self.training,
            "Device": self.device,
            "Logging": self.logging
        }
        
        for section_name, section_config in sections.items():
            print(f"\n{section_name}:")
            print("-" * 40)
            for field, value in section_config.model_dump().items():
                print(f"  {field}: {value}")
        
        print("\n" + "=" * 80)
    
    def validate_config(self) -> bool:
        """
        Validate the entire configuration.
        
        Returns:
            bool: True if configuration is valid
            
        Raises:
            ValueError: If configuration is invalid
        """
        # Validate paths exist if needed
        if not Path(self.data.data_root).exists():
            raise ValueError(f"Data root directory does not exist: {self.data.data_root}")
        
        return True


def load_config(yaml_path: str = "./config/config.yaml") -> SLDetectorConfig:
    """
    Load Sign Language Detector configuration from YAML file.
    
    Args:
        yaml_path: Path to the YAML configuration file
        
    Returns:
        SLDetectorConfig: Validated configuration object
    """
    return SLDetectorConfig.from_yaml(yaml_path)


if __name__ == "__main__":
    # Example usage
    config = load_config("config/config.yaml")
    config.print_config()
    print(f"\nDevice: {config.get_device()}")
    print(f"Best checkpoint path: {config.get_checkpoint_path(is_best=True)}")
    
    # Create necessary directories
    config.create_directories()
    print("\nAll directories created successfully!")