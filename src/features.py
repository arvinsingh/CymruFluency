import numpy as np

import torch
import librosa
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC


def signal_static_features(audio_data, sample_rates=16000):
    X, sample_rate = audio_data, sample_rates
    if X.ndim > 1:
        X = X[:, 0]
    X = X.T

    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=20).T, axis=0)
    rmse = np.mean(librosa.feature.rms(y=X).T, axis=0)
    spectral_flux = np.mean(librosa.onset.onset_strength(y=X, sr=sample_rate).T, axis=0)
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=X).T, axis=0)

    return mfccs, rmse, spectral_flux, zcr


def signal_dynamic_features(audio_data, sample_rates=16000):
    X, sample_rate = audio_data, sample_rates
    if X.ndim > 1:
        X = X[:, 0]  # Convert to mono if stereo
    
    # Compute features without averaging
    mfccs = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=20)
    rmse = librosa.feature.rms(y=X)
    spectral_flux = librosa.onset.onset_strength(y=X, sr=sample_rate)
    zcr = librosa.feature.zero_crossing_rate(y=X)
    
    spectral_flux = spectral_flux.reshape(1, -1)
    features = np.vstack((mfccs, rmse, spectral_flux, zcr))
    features = features.T  # Shape (126, 23)
    
    return features


def audio_features(audio_data, sample_rates=16000, static=True):
    if static:
        all_features = np.zeros((len(audio_data), 23))
    else:
        all_features = np.zeros((len(audio_data), 126, 23))
    for i, audio in enumerate(audio_data):
        if static:
            mfccs, rmse, spectral_flux, zcr = signal_static_features(audio, sample_rates)
            all_features[i] = np.hstack([mfccs, rmse, spectral_flux, zcr])
        else:
            features = signal_dynamic_features(audio, sample_rates)
            all_features[i] = features
    return all_features


def wav2vec_features(audio_data, sample_rate=16000):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load the processor and model
    processor = Wav2Vec2Processor.from_pretrained("techiaith/wav2vec2-xlsr-ft-cy")
    model = Wav2Vec2ForCTC.from_pretrained(
        "techiaith/wav2vec2-xlsr-ft-cy", 
        output_hidden_states=True
    )
    model.to(device)
    
    all_features = []
    
    for i, signal in enumerate(audio_data):
        # Process the input signal
        inputs = processor(
            signal, 
            sampling_rate=sample_rate, 
            return_tensors="pt", 
            padding=True
        )
        # Move inputs to device
        inputs = {key: value.to(device) for key, value in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Extract the hidden states (features)
        # outputs.hidden_states is a tuple of all hidden states from each layer
        hidden_states = outputs.hidden_states[-1]  # Last layer's hidden state
        hidden_states = hidden_states.squeeze()  # Remove batch dimension
        features = hidden_states.cpu().numpy()
        
        all_features.append(features)
    
    return all_features



def get_mouth_landmarks(landmarks_data):
    # points 48 to 68 correspond to the mouth contour
    # Expecting shape (3, 20)
    mouth_contour_points = landmarks_data[..., 48:68]
    return mouth_contour_points


def align_landmarks(landmarks_data):
    # Align landmarks to the center of the mouth
    mouth_landmarks = get_mouth_landmarks(landmarks_data)
    mouth_center = np.mean(mouth_landmarks, axis=0)
    aligned_landmarks = landmarks_data['landmarks']['points'] - mouth_center
    return aligned_landmarks


def align_face_frames(face_landmarks, nose_index=37, left_eye_index=20, right_eye_index=23, scale=True):
    # Align face landmarks across frames
    num_frames, num_channels, num_landmarks = face_landmarks.shape
    aligned_landmarks = np.zeros_like(face_landmarks)

    for i in range(num_frames):
        frame = face_landmarks[i] 

        # Step 1: Translate so nose is at origin
        nose_point = frame[:, nose_index]
        translated_frame = frame - nose_point[:, np.newaxis]

        # Step 2: Calculate angle between eyes
        left_eye = translated_frame[:, left_eye_index]
        right_eye = translated_frame[:, right_eye_index]
        eye_vector = right_eye - left_eye

        # Calculate rotation angle
        angle = np.arctan2(eye_vector[1], eye_vector[0])

        # Create rotation matrix
        cos_theta = np.cos(-angle)
        sin_theta = np.sin(-angle)
        rotation_matrix = np.array([
            [cos_theta, -sin_theta],
            [sin_theta,  cos_theta]
        ])

        # Apply rotation to 2D coordinates
        if num_channels >= 2:
            rotated_xy = rotation_matrix @ translated_frame[:2, :]
            if num_channels == 2:
                rotated_frame = rotated_xy
            else:
                # Combine rotated x, y with original z
                rotated_frame = np.vstack((rotated_xy, translated_frame[2:, :]))
        else:
            raise ValueError("Number of channels must be at least 2.")

        # Step 3: Scale landmarks (optional)
        if scale:
            inter_eye_distance = np.linalg.norm(right_eye - left_eye)
            if inter_eye_distance == 0:
                print(f"Warning: Inter-eye distance is zero at frame {i}. Skipping scaling.")
                scale_factor = 1.0
            else:
                scale_factor = 1.0 / inter_eye_distance
            scaled_frame = rotated_frame * scale_factor
        else:
            scaled_frame = rotated_frame

        # Store the aligned frame
        aligned_landmarks[i] = scaled_frame

    return aligned_landmarks


def align_all_faces(landmarks, nose_index=37, left_eye_index=20, right_eye_index=23, scale=False):

    num_faces, num_frames, num_channels, num_landmarks = landmarks.shape
    aligned_landmarks = np.zeros_like(landmarks)

    for face_idx in range(num_faces):
        face_landmarks = landmarks[face_idx]
        # Align the frames of this face
        aligned_face_landmarks = align_face_frames(
            face_landmarks,
            nose_index,
            left_eye_index,
            right_eye_index,
            scale=scale,
        )
        aligned_landmarks[face_idx] = aligned_face_landmarks

    return aligned_landmarks


def align_face_frames_3d(face_landmarks, nose_index=37, left_eye_index=20, right_eye_index=23, chin_index=8, scale=True):

    num_frames, num_channels, num_landmarks = face_landmarks.shape
    aligned_landmarks = np.zeros_like(face_landmarks)

    for i in range(num_frames):
        frame = face_landmarks[i]  # Shape: (3, num_landmarks)

        # Step 1: Translate so nose is at origin
        nose_point = frame[:, nose_index]  # Shape: (3,)
        translated_frame = frame - nose_point[:, np.newaxis]

        # Step 2: Compute rotation matrix to align face
        # Define source vectors based on facial landmarks
        left_eye = translated_frame[:, left_eye_index]
        right_eye = translated_frame[:, right_eye_index]
        chin = translated_frame[:, chin_index]

        # Compute the vectors between key points
        eye_vector = right_eye - left_eye  # Vector between eyes
        eye_midpoint = (left_eye + right_eye) / 2
        nose_to_chin = chin - nose_point   # Vector from nose to chin

        # Create a local coordinate system
        # x-axis: eye_vector (from left eye to right eye)
        # y-axis: perpendicular to eye_vector and nose_to_chin
        # z-axis: perpendicular to x and y axes

        x_axis = eye_vector / np.linalg.norm(eye_vector)
        z_axis = np.cross(eye_vector, nose_to_chin)
        z_axis = z_axis / np.linalg.norm(z_axis)
        y_axis = np.cross(z_axis, x_axis)
        y_axis = y_axis / np.linalg.norm(y_axis)

        # Assemble rotation matrix from axes
        rotation_matrix = np.vstack([x_axis, y_axis, z_axis]).T  # Shape: (3, 3)

        # Apply rotation to all landmarks
        rotated_frame = rotation_matrix.T @ translated_frame  # Transpose to invert rotation

        # Step 3: Scale landmarks (optional)
        if scale:
            # Use inter-eye distance as scale reference
            inter_eye_distance = np.linalg.norm(rotated_frame[:, left_eye_index] - rotated_frame[:, right_eye_index])
            if inter_eye_distance == 0:
                print(f"Warning: Inter-eye distance is zero at frame {i}. Skipping scaling.")
                scale_factor = 1.0
            else:
                scale_factor = 1.0 / inter_eye_distance
            scaled_frame = rotated_frame * scale_factor
        else:
            scaled_frame = rotated_frame

        # Store the aligned frame
        aligned_landmarks[i] = scaled_frame

    return aligned_landmarks

def align_all_faces_3d(landmarks, nose_index=37, left_eye_index=20, right_eye_index=23, chin_index=8, scale=True):

    num_faces, num_frames, num_channels, num_landmarks = landmarks.shape
    aligned_landmarks = np.zeros_like(landmarks)

    for face_idx in range(num_faces):
        face_landmarks = landmarks[face_idx]
        # Align the frames of this face
        aligned_face_landmarks = align_face_frames_3d(face_landmarks, nose_index, left_eye_index, right_eye_index, chin_index, scale=scale)
        aligned_landmarks[face_idx] = aligned_face_landmarks

    return aligned_landmarks


def adjacency_matrix(connectivity_dict=None):
    if connectivity_dict is None:
        connectivity_dict = landmark_connectivity_dictionary()
        num_landmarks = 68
    else:
        num_landmarks = 20

    adj_matrix = np.zeros((num_landmarks, num_landmarks))

    for segment in connectivity_dict.values():
        for i in range(len(segment) - 1):
            start = segment[i]
            end = segment[i + 1]
            adj_matrix[start, end] = 1
            adj_matrix[end, start] = 1

    return torch.tensor(adj_matrix, dtype=torch.float32)


def landmark_connectivity_dictionary():
    segments = {
        'Jaw': list(range(0, 17)),
        'Right Eye': list(range(17, 23)) + [17],
        'Left Eye': list(range(23, 29)) + [23],
        'Right Eyebrow': list(range(29, 34)),
        'Nose': list(range(34, 38)),
        'Under Nose': list(range(38, 43)),
        'Left Eyebrow': list(range(43, 48)),
        'Outer Lip': list(range(48, 60)) + [48],
        'Inner Lip': list(range(60, 68)) + [60],
    }

    return segments

def mouth_connectivity_dictionary():
    segments = {
        'Outer Lip': list(range(0, 12)) + [0],
        'Inner Lip': list(range(12, 20)) + [12],
    }

    return segments