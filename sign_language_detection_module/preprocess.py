import os
import shutil
from tqdm import tqdm
import cv2
import numpy as np
import pandas as pd


class SLDatasetPreprocessor:
    def __init__(self, frames_per_second=5, max_frames_per_video=37, 
                 trim_start_portion=0.15, trim_end_portion=0.85):
        self.frames_per_second = frames_per_second
        self.max_frames_per_video = max_frames_per_video
        self.trim_start_portion = trim_start_portion
        self.trim_end_portion = trim_end_portion
    
    def copy_retained_classes(self, source_dir, target_dir, retained_classes_csv):
        os.makedirs(target_dir, exist_ok=True)
        
        df_retained_classes = pd.read_csv(retained_classes_csv)
        classes_to_keep = set(df_retained_classes['class_name'].unique())
        
        for folder in tqdm(os.listdir(source_dir)):
            src_path = os.path.join(source_dir, folder)
            
            if folder in classes_to_keep:
                dst_path = os.path.join(target_dir, folder)
                shutil.copytree(src_path, dst_path)
    
    def extract_frames_from_videos(self, videos_dir, output_img_dir):
        os.makedirs(output_img_dir, exist_ok=True)
        
        class_list = [c for c in os.listdir(videos_dir) if os.path.isdir(os.path.join(videos_dir, c))]
        
        bad_videos = []
        
        for cls in tqdm(class_list, desc="Processing SL Classes", ncols=100):
            class_path = os.path.join(videos_dir, cls)
            out_class_dir = os.path.join(output_img_dir, cls)
            os.makedirs(out_class_dir, exist_ok=True)
            
            videos = [v for v in os.listdir(class_path) if v.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))]
            
            for video in videos:
                video_path = os.path.join(class_path, video)
                
                cap = cv2.VideoCapture(video_path)
                
                if not cap.isOpened():
                    bad_videos.append(video_path)
                    continue
                
                fps = cap.get(cv2.CAP_PROP_FPS)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                
                if fps <= 1 or total_frames <= 1:
                    bad_videos.append(video_path)
                    cap.release()
                    continue
                
                frame_interval = max(int(fps / self.frames_per_second), 1)
                
                frames = []
                frame_num = 0
                
                while True:
                    ret, frame = cap.read()
                    
                    if not ret:
                        break
                    
                    if frame_num % frame_interval == 0:
                        frames.append(frame)
                    
                    frame_num += 1
                
                cap.release()
                
                if len(frames) == 0:
                    bad_videos.append(video_path)
                    continue
                
                if len(frames) > 4:
                    total = len(frames)
                    start = int(total * self.trim_start_portion)
                    end = int(total * self.trim_end_portion)
                    frames = frames[start:end]
                
                if len(frames) == 0:
                    bad_videos.append(video_path)
                    continue
                
                indices = np.linspace(0, len(frames) - 1, self.max_frames_per_video).astype(int)
                frames = [frames[i] for i in indices]
                
                video_name = os.path.splitext(video)[0]
                out_video_dir = os.path.join(out_class_dir, video_name)
                os.makedirs(out_video_dir, exist_ok=True)
                
                for i, frame in enumerate(frames):
                    cv2.imwrite(os.path.join(out_video_dir, f"{video_name}_frame{i:04d}.jpg"), frame)
        
        print("Frame extraction complete.")
        print("\nThe following videos were corrupted or unreadable:")
        for b in bad_videos:
            print(" -", b)
        
        return bad_videos


class JesterDatasetPreprocessor:
    def __init__(self):
        pass
    
    def copy_classes(self, jester_csv, jester_train_path, output_path, target_classes):
        df_jester_train = pd.read_csv(jester_csv)
        
        df_no_class = df_jester_train[df_jester_train['label'].isin(target_classes)].reset_index(drop=True)
        
        video_ids = df_no_class["video_id"].astype(str).tolist()
        
        train_folders = set(os.listdir(jester_train_path))
        
        matching_folders = [vid for vid in video_ids if vid in train_folders]
        
        os.makedirs(output_path, exist_ok=True)
        
        for vid in matching_folders:
            src = os.path.join(jester_train_path, vid)
            dst = os.path.join(output_path, vid)
            shutil.copytree(src, dst)
        
        print(f"Copied {len(matching_folders)} folders.")
        
        return matching_folders
    
    def create_archive(self, output_path):
        return shutil.make_archive(output_path, 'zip', output_path)