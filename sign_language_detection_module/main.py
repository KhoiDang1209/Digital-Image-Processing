import argparse
import sys
import torch
import random
import numpy as np

from config.config import load_config
from train import Trainer
from transform import VideoToNpyConverter, FramesToNpyConverter
from preprocess import SLDatasetPreprocessor, JesterDatasetPreprocessor


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_model(args):
    config = load_config(args.config)
    set_seed(config.seed)
    
    print(f"Starting training with config: {args.config}")
    config.print_config()
    
    trainer = Trainer(
        config=config,
        use_cache=args.use_cache,
        fn_penalty_factor=args.fn_penalty
    )
    
    print(f"\nModel parameters: {sum(p.numel() for p in trainer.model.parameters()):,}")
    
    test_acc, test_recall = trainer.train_model()
    
    print(f"\n{'='*60}")
    print(f"Training Complete!")
    print(f"{'='*60}")
    print(f"Final Test Accuracy: {test_acc:.2f}%")
    print(f"YES Class Recall: {test_recall*100:.2f}%")
    print(f"Best Val YES Recall: {trainer.best_val_recall*100:.2f}%")
    print(f"{'='*60}")


def test_model(args):
    config = load_config(args.config)
    set_seed(config.seed)
    
    print(f"Testing model: {args.checkpoint}")
    
    trainer = Trainer(
        config=config,
        use_cache=args.use_cache,
        fn_penalty_factor=args.fn_penalty
    )
    
    checkpoint = torch.load(args.checkpoint, map_location=config.get_device())
    trainer.model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown') + 1}")
    
    test_acc, test_recall = trainer.test()
    
    print(f"\nTest Accuracy: {test_acc:.2f}%")
    print(f"YES Class Recall: {test_recall*100:.2f}%")


def preprocess_sl(args):
    print("Preprocessing SL dataset...")
    
    preprocessor = SLDatasetPreprocessor(
        frames_per_second=args.fps,
        max_frames_per_video=args.max_frames,
        trim_start_portion=args.trim_start,
        trim_end_portion=args.trim_end
    )
    
    if args.copy_classes:
        print(f"Copying retained classes from {args.source_dir} to {args.target_dir}")
        preprocessor.copy_retained_classes(
            args.source_dir,
            args.target_dir,
            args.retained_csv
        )
    
    if args.extract_frames:
        print(f"Extracting frames from {args.videos_dir} to {args.output_dir}")
        bad_videos = preprocessor.extract_frames_from_videos(
            args.videos_dir,
            args.output_dir
        )
        print(f"Completed. {len(bad_videos)} videos had errors.")


def preprocess_jester(args):
    print("Preprocessing Jester dataset...")
    
    preprocessor = JesterDatasetPreprocessor()
    
    target_classes = args.target_classes.split(',') if args.target_classes else ['Doing other things', 'No gesture']
    
    matching_folders = preprocessor.copy_classes(
        args.jester_csv,
        args.jester_train_path,
        args.output_path,
        target_classes
    )
    
    print(f"Copied {len(matching_folders)} folders")
    
    if args.create_archive:
        print("Creating archive...")
        archive_path = preprocessor.create_archive(args.output_path)
        print(f"Archive created: {archive_path}")


def convert_videos(args):
    print(f"Converting videos from {args.input_dir} to numpy arrays...")
    
    converter = VideoToNpyConverter(
        seq_len=args.seq_len,
        max_hands=args.max_hands
    )
    
    import os
    from tqdm import tqdm
    
    all_videos = []
    for cls in os.listdir(args.input_dir):
        cls_path = os.path.join(args.input_dir, cls)
        if not os.path.isdir(cls_path):
            continue
        
        out_dir = os.path.join(args.output_dir, cls)
        os.makedirs(out_dir, exist_ok=True)
        
        for vid in os.listdir(cls_path):
            if vid.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
                full_path = os.path.join(cls_path, vid)
                all_videos.append((full_path, out_dir))
    
    print(f"Processing {len(all_videos)} videos...")
    
    for video_path, out_dir in tqdm(all_videos, desc="Converting videos"):
        try:
            converter.save(video_path, out_dir)
        except Exception as e:
            print(f"\nERROR: {video_path} - {e}")
    
    print("Video conversion complete!")


def convert_frames(args):
    print(f"Converting frames from {args.input_dir} to numpy arrays...")
    
    converter = FramesToNpyConverter(
        seq_len=args.seq_len,
        max_hands=args.max_hands
    )
    
    import os
    from tqdm import tqdm
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    for video_folder in tqdm(os.listdir(args.input_dir)):
        frames_dir = os.path.join(args.input_dir, video_folder)
        if not os.path.isdir(frames_dir):
            continue
        
        try:
            converter.save(frames_dir, args.output_dir)
        except Exception as e:
            print(f"ERROR: {frames_dir} - {e}")
    
    print("Frame conversion complete!")


def main():
    parser = argparse.ArgumentParser(description='Sign Language Detection Module')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument('--config', type=str, default='config/config.yaml', help='Path to config file')
    train_parser.add_argument('--use-cache', action='store_true', default=True, help='Use cached dataset')
    train_parser.add_argument('--no-cache', dest='use_cache', action='store_false', help='Do not use cached dataset')
    train_parser.add_argument('--fn-penalty', type=float, default=2.0, help='False negative penalty factor')
    
    test_parser = subparsers.add_parser('test', help='Test the model')
    test_parser.add_argument('--config', type=str, default='config/config.yaml', help='Path to config file')
    test_parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    test_parser.add_argument('--use-cache', action='store_true', default=True, help='Use cached dataset')
    test_parser.add_argument('--no-cache', dest='use_cache', action='store_false', help='Do not use cached dataset')
    test_parser.add_argument('--fn-penalty', type=float, default=2.0, help='False negative penalty factor')
    
    preprocess_sl_parser = subparsers.add_parser('preprocess-sl', help='Preprocess SL dataset')
    preprocess_sl_parser.add_argument('--copy-classes', action='store_true', help='Copy retained classes')
    preprocess_sl_parser.add_argument('--source-dir', type=str, help='Source directory for classes')
    preprocess_sl_parser.add_argument('--target-dir', type=str, help='Target directory for classes')
    preprocess_sl_parser.add_argument('--retained-csv', type=str, help='CSV file with retained classes')
    preprocess_sl_parser.add_argument('--extract-frames', action='store_true', help='Extract frames from videos')
    preprocess_sl_parser.add_argument('--videos-dir', type=str, help='Directory containing videos')
    preprocess_sl_parser.add_argument('--output-dir', type=str, help='Output directory for frames')
    preprocess_sl_parser.add_argument('--fps', type=int, default=5, help='Frames per second')
    preprocess_sl_parser.add_argument('--max-frames', type=int, default=37, help='Max frames per video')
    preprocess_sl_parser.add_argument('--trim-start', type=float, default=0.15, help='Trim start portion')
    preprocess_sl_parser.add_argument('--trim-end', type=float, default=0.85, help='Trim end portion')
    
    preprocess_jester_parser = subparsers.add_parser('preprocess-jester', help='Preprocess Jester dataset')
    preprocess_jester_parser.add_argument('--jester-csv', type=str, required=True, help='Jester CSV file')
    preprocess_jester_parser.add_argument('--jester-train-path', type=str, required=True, help='Jester train path')
    preprocess_jester_parser.add_argument('--output-path', type=str, required=True, help='Output path')
    preprocess_jester_parser.add_argument('--target-classes', type=str, help='Target classes (comma-separated)')
    preprocess_jester_parser.add_argument('--create-archive', action='store_true', help='Create zip archive')
    
    convert_videos_parser = subparsers.add_parser('convert-videos', help='Convert videos to numpy arrays')
    convert_videos_parser.add_argument('--input-dir', type=str, required=True, help='Input video directory')
    convert_videos_parser.add_argument('--output-dir', type=str, required=True, help='Output numpy directory')
    convert_videos_parser.add_argument('--seq-len', type=int, default=37, help='Sequence length')
    convert_videos_parser.add_argument('--max-hands', type=int, default=2, help='Maximum number of hands')
    
    convert_frames_parser = subparsers.add_parser('convert-frames', help='Convert frames to numpy arrays')
    convert_frames_parser.add_argument('--input-dir', type=str, required=True, help='Input frames directory')
    convert_frames_parser.add_argument('--output-dir', type=str, required=True, help='Output numpy directory')
    convert_frames_parser.add_argument('--seq-len', type=int, default=37, help='Sequence length')
    convert_frames_parser.add_argument('--max-hands', type=int, default=2, help='Maximum number of hands')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        train_model(args)
    elif args.command == 'test':
        test_model(args)
    elif args.command == 'preprocess-sl':
        preprocess_sl(args)
    elif args.command == 'preprocess-jester':
        preprocess_jester(args)
    elif args.command == 'convert-videos':
        convert_videos(args)
    elif args.command == 'convert-frames':
        convert_frames(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
