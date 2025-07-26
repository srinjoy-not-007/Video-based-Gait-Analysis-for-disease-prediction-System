import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
import os
import time
from tqdm import tqdm
from scipy.signal import savgol_filter, find_peaks

DATASET_PATH = r"D:\GUI Gait parameters\videos\KOA-PD-NM\dataset" 
OUTPUT_DATASET_CSV = 'gait_features_with_labels.csv'
FPS = 30  


mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)



def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_num = 0
    data = []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"    Frames to process: {total_frames}")

    with tqdm(total=total_frames, desc=f"    Processing {os.path.basename(video_path)}") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = pose.process(image)

            if result.pose_landmarks:
                landmarks = result.pose_landmarks.landmark

                hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
                knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE]
                ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE]

                data.append([
                    frame_num,
                    hip.x, hip.y, hip.z,
                    knee.x, knee.y, knee.z,
                    ankle.x, ankle.y, ankle.z
                ])

            frame_num += 1
            pbar.update(1)

    cap.release()

    if len(data) == 0:
        print("    No pose detected, skipping video.")
        return None

    df = pd.DataFrame(data, columns=[
        'frame',
        'hip_x', 'hip_y', 'hip_z',
        'knee_x', 'knee_y', 'knee_z',
        'ankle_x', 'ankle_y', 'ankle_z'
    ])

    return df

def filter_keypoints(df):
    for col in df.columns[1:]:
        df[col] = savgol_filter(df[col], window_length=7, polyorder=3)
    return df

def detect_heel_strikes(df):
    ankle_y = df['ankle_y']
    peaks, _ = find_peaks(-ankle_y, distance=20)
    return peaks

def calculate_gait_parameters(df, heel_strikes):
    if len(heel_strikes) < 2:
        print("Not enough heel strikes detected, skipping.")
        return None

    step_lengths = []
    for i in range(1, len(heel_strikes)):
        idx_prev = heel_strikes[i-1]
        idx_curr = heel_strikes[i]

        hip_x_prev = df.loc[idx_prev, 'hip_x']
        hip_x_curr = df.loc[idx_curr, 'hip_x']

        step_length = abs(hip_x_curr - hip_x_prev)
        step_lengths.append(step_length)

    cadence = (len(heel_strikes) / (df['frame'].iloc[-1] / FPS)) * 60

    gait_features = {
        'step_length': np.mean(step_lengths),
        'stride_length': np.mean(step_lengths) * 2,
        'cadence': cadence,
        'hip_rom': df['hip_y'].max() - df['hip_y'].min(),
        'knee_rom': df['knee_y'].max() - df['knee_y'].min(),
        'ankle_rom': df['ankle_y'].max() - df['ankle_y'].min()
    }

    return gait_features

#MAIN PIPELINE
start_time = time.time()
dataset_features = []

label_folders = [f for f in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, f))]

for label_folder in label_folders:
    label_path = os.path.join(DATASET_PATH, label_folder)
    print(f"\nProcessing label: {label_folder}")

    video_files = [f for f in os.listdir(label_path) if f.lower().endswith(('.mp4', '.mov'))]


    for video_file in video_files:
        video_path = os.path.join(label_path, video_file)
        print(f"\n  Processing video: {video_file}")

        video_start = time.time()
        df_keypoints = process_video(video_path)

        if df_keypoints is None:
            continue

        df_filtered = filter_keypoints(df_keypoints)
        heel_strikes = detect_heel_strikes(df_filtered)
        gait_params = calculate_gait_parameters(df_filtered, heel_strikes)

        if gait_params is None:
            continue

        gait_params['label'] = label_folder
        dataset_features.append(gait_params)

        video_end = time.time()
        print(f"  Finished processing {video_file} in {video_end - video_start:.2f} seconds")

df_dataset = pd.DataFrame(dataset_features)
df_dataset.to_csv(OUTPUT_DATASET_CSV, index=False)
end_time = time.time()

print(f"\nDataset saved to {OUTPUT_DATASET_CSV}")
print(f" Total processing time: {(end_time - start_time)/60:.2f} minutes")
