import h5py
import numpy as np
import time
import os
import cv2
import threading
from collections import deque

class DataRecorder:
    def __init__(self, save_dir="data_collection", max_memory_steps=300):
        self.save_dir = save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        self.recording = False
        self.buffer_qpos = []
        self.buffer_qvel = []
        self.buffer_action = []
        self.buffer_images = {} 
        self.lock = threading.Lock()
        
        self.start_time = 0.0
        self.episode_idx = 0
        
        # ACT/ALOHA usually expects 30Hz or 50Hz.
        # We will collect whatever is fed to us.
        
    def start_recording(self):
        with self.lock:
            self.recording = True
            self.buffer_qpos = []
            self.buffer_qvel = []
            self.buffer_action = []
            self.buffer_images = {}
            self.start_time = time.time()
            print(f"[RECORDER] Started Episode {self.episode_idx}")

    def stop_recording(self):
        with self.lock:
            if not self.recording: return
            self.recording = False
            duration = time.time() - self.start_time
            print(f"[RECORDER] Stopped. Duration: {duration:.2f}s. Saving...")
            self.save_episode()
            self.episode_idx += 1

    def add_frame(self, qpos, qvel, action, images: dict):
        """
        qpos: np.array (7,)
        qvel: np.array (7,)
        action: np.array (7,)
        images: dict of name -> np.array (H, W, 3)
        """
        with self.lock:
            if not self.recording: return
            
            self.buffer_qpos.append(qpos)
            self.buffer_qvel.append(qvel)
            self.buffer_action.append(action)
            
            for cam_name, img in images.items():
                if cam_name not in self.buffer_images:
                    self.buffer_images[cam_name] = []
                self.buffer_images[cam_name].append(img)

    def save_episode(self):
        if len(self.buffer_qpos) == 0:
            print("[RECORDER] Buffer empty, nothing to save.")
            return

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.save_dir, f"episode_{self.episode_idx}_{timestamp}.hdf5")
        
        try:
            with h5py.File(filename, 'w') as f:
                f.create_dataset('action', data=np.array(self.buffer_action))
                
                obs_grp = f.create_group('observations')
                obs_grp.create_dataset('qpos', data=np.array(self.buffer_qpos))
                obs_grp.create_dataset('qvel', data=np.array(self.buffer_qvel))
                
                img_grp = obs_grp.create_group('images')
                for cam_name, frames in self.buffer_images.items():
                    img_grp.create_dataset(cam_name, data=np.array(frames), compression='gzip')
                    
            print(f"[RECORDER] Saved {filename} ({len(self.buffer_qpos)} steps)")
        except Exception as e:
            print(f"[RECORDER] Error saving file: {e}")
