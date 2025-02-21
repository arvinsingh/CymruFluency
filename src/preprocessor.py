import os
import glob
import numpy as np

import librosa
import torch
import py7zr
import json
import shutil


def get_landmarks_files(data_path):
    file_list = []
    for root, _, files in os.walk(data_path):
        for file in files:
            if file.endswith(".7z"):
                file_list.append(os.path.join(root, file))
    return file_list


def extract_contour_points(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
        landmarks = data['landmarks']['points']

    return torch.tensor(landmarks)


def get_sequences(landmarks_path, max_frames=250):
    sequences = torch.zeros((max_frames, 3, 68))
    with py7zr.SevenZipFile(landmarks_path, mode='r') as archive:
        archive.extractall('temp')
        frame_names = [name for name in archive.getnames() if name.endswith(".ljson")]
        frame_names.sort()  # processing frames in order
        for i, name in enumerate(frame_names):
            if i >= max_frames:
                break
            try:
                data = extract_contour_points(os.path.join('temp', name))
                sequences[i] = data.T
            except Exception as e:
                print(f"[Error][{name}] Landmark extraction: {e}")
                continue
        shutil.rmtree('temp')
    return sequences


def normalize_signal(signal):
    mean = np.mean(signal)
    std = np.std(signal)
    
    # Handle case where std is zero to avoid division by zero
    if std == 0:
        std = 1e-8
    
    normalized_signal = (signal - mean) / std
    return normalized_signal



def parse_combined_files(audio_parent_dir, audio_sub_dirs, landmarks_root, landmarks_sub_dirs,
                         SAVE_DIR, max_frames=250, audio_ext='wav', target_sr=16000):
    """
    Build dictionaries from both modalities using the file's base name (without extension)
    and then produce combined arrays that are aligned by matching base name.

    Parameters:
      audio_parent_dir : root directory for audio files
      audio_sub_dirs   : list of subdirectories for audio files
      landmarks_root   : root directory for landmarks (e.g. where face data is stored)
      landmarks_sub_dirs: list of susbdirectories for landmarks files
      max_frames       : maximum number of frames to process for landmarks
      audio_ext        : audio file extension (without dot)

    Returns:
      combined_audio     : np.array of audio features (one per file)
      combined_landmarks : np.array of landmark sequences (one per file)
      labels             : np.array of labels (from audio sub_dirs)
                           Note: labels are taken from the audio directories.
    """
    print("Extracting features from audio and landmarks...")
    audio_dict = {}
    audio_labels = {}
    # Process audio files
    for label, sub_dir in enumerate(audio_sub_dirs):
        print(f"Processing {sub_dir}...")
        audio_path = os.path.join(audio_parent_dir, sub_dir)
        pattern = os.path.join(audio_path, f"*.{audio_ext}")
        files = glob.glob(pattern)
        total_files = len(files)
        for idx, file in enumerate(files):
            base = os.path.splitext(os.path.basename(file))[0]
            try:
                signal, original_sr = librosa.load(file, sr=None)
                # resample if necessary
                if original_sr != target_sr:
                    signal = librosa.resample(signal, orig_sr=original_sr, target_sr=target_sr)
                # normalize signal
                signal = normalize_signal(signal)
            except Exception as e:
                print(f"[Error][{file}] Audio feature extraction: {e}")
                continue
            audio_dict[base] = signal
            audio_labels[base] = label
            progress = (idx + 1) / total_files * 100
            print(f"Processing {sub_dir}: {idx + 1}/{total_files} files ({progress:.2f}% done)", end='\r')
        print(f"\nExtracted features from {sub_dir}, done")
    
    print("Audio features extracted")

    landmark_dict = {}
    landmark_labels = {}
    # Process landmark files
    for label, sub_dir in enumerate(landmarks_sub_dirs):
        print(f"Processing {sub_dir}...")
        dir_path = os.path.join(landmarks_root, sub_dir)
        files = get_landmarks_files(dir_path)
        total_files = len(files)
        for idx, file in enumerate(files):
            base = os.path.splitext(os.path.basename(file))[0]
            try:
                sequence = get_sequences(file, max_frames)
            except Exception as e:
                print(f"[Error][{file}] Landmark extraction: {e}")
                continue
            # Convert sequence to numpy
            landmark_dict[base] = sequence.numpy()
            landmark_labels[base] = label
            progress = (idx + 1) / total_files * 100
            print(f"Processing {sub_dir}: {idx + 1}/{total_files} files ({progress:.2f}% done)", end='\r')
        print(f"\nExtracted landmarks from {sub_dir}, done")
    
    print("Landmark sequences extracted")

    # Get matching base names from both modalities.
    common_keys = sorted(set(audio_dict.keys()) & set(landmark_dict.keys()))
    combined_audio = []
    combined_landmarks = []
    labels = []
    for key in common_keys:
        combined_audio.append(audio_dict[key])
        combined_landmarks.append(landmark_dict[key])
        labels.append(audio_labels[key]) # use audio labels
    
    # pad audio signals to the same length
    max_len = max([len(audio) for audio in combined_audio])
    combined_audio = [np.pad(audio, (0, max_len - len(audio))) for audio in combined_audio]
    combined_audio = np.array(combined_audio)
    combined_landmarks = np.array(combined_landmarks)
    labels = np.array(labels)
    save_processed_dataset(combined_audio, combined_landmarks, labels, SAVE_DIR)


def save_processed_dataset(feature_1, feature_2, labels, save_path):
    np.save(save_path + 'audios', feature_1)
    np.save(save_path + 'landmarks', feature_2)
    np.save(save_path + 'labels', labels)
    print(f"Processed dataset saved to {save_path}")


def load_processed_dataset(load_path):
    feature_1 = np.load(load_path + 'audios.npy')
    feature_2 = np.load(load_path + 'landmarks.npy')
    labels = np.load(load_path + 'labels.npy')
    return feature_1, feature_2, labels


if __name__ == '__main__':
    SAVE_DIR = 'data/dataset/'
    AUDIO_PARENT_DIR = 'data/raw/Audio'
    LANDMARK_PARENT_DIR = 'data/raw/Landmarks'
    CLASSES = ['High', 'Low']
    print("Creating Dataset")
    parse_combined_files(AUDIO_PARENT_DIR, CLASSES, LANDMARK_PARENT_DIR, CLASSES, SAVE_DIR, audio_ext='wav')
    print("Dataset Created")
