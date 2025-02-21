from torch.utils.data import Dataset
import torch
import numpy as np


class CymruFluencyDataset(Dataset):
    def __init__(self, audio_data, landmark_data, labels):
        """
        Parameters:
          audio_data: numpy array of shape (327, 23)
          landmark_data: numpy array of shape (327, 250, 3, 20)
          labels: numpy array of shape (327,) with integer labels (0 or 1)
        """
        self.audio_data = torch.tensor(audio_data, dtype=torch.float32)
        self.landmark_data = torch.tensor(landmark_data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        
    def __len__(self):
        return len(self.audio_data)
    
    def __getitem__(self, idx):
        return self.audio_data[idx], self.landmark_data[idx], self.labels[idx]


class SiameseCymruFluencyDataset(Dataset):
    def __init__(self, audio_data, landmark_data, labels, transform=None):
        """
        Initializes the Siamese dataset.
        
        Parameters:
          audio_data: Array of shape (N, ...) containing audio features. This can be either a
                      2D tensor (N, feature_dim) or a 3D tensor (N, seq_len, feature_dim).
          landmark_data: Array of shape (N, T, C, V) where T is the number of frames,
                         C is the number of channels (e.g., 3 for x, y, z), and V is the number
                         of landmarks (e.g., 20 for mouth landmarks).
          labels: Array of shape (N,) with class labels.
          transform: Optional transformation function applied to the data.
        """
        # Convert to tensors if needed.
        if not torch.is_tensor(audio_data):
            self.audio_data = torch.tensor(audio_data, dtype=torch.float32)
        else:
            self.audio_data = audio_data

        if not torch.is_tensor(landmark_data):
            self.landmark_data = torch.tensor(landmark_data, dtype=torch.float32)
        else:
            self.landmark_data = landmark_data

        if not torch.is_tensor(labels):
            self.labels = torch.tensor(labels, dtype=torch.long)
        else:
            self.labels = labels

        self.transform = transform
        self.num_samples = len(self.labels)
        
        # Build a dictionary mapping each label to a list of indices.
        self.label_to_indices = {}
        for i, label in enumerate(self.labels):
            lbl = int(label.item())
            if lbl not in self.label_to_indices:
                self.label_to_indices[lbl] = []
            self.label_to_indices[lbl].append(i)
        
    def __len__(self):
        # Define the length of the dataset as the number of pairs to generate.
        return self.num_samples
    
    def __getitem__(self, index):
        # Use the given index as the anchor.
        anchor_idx = index % self.num_samples
        anchor_audio = self.audio_data[anchor_idx]
        anchor_landmarks = self.landmark_data[anchor_idx]
        anchor_label = int(self.labels[anchor_idx].item())
        
        # Randomly decide whether to form a positive (same class) or negative (different class) pair.
        same_class = np.random.choice([True, False])
        if same_class:
            # Select a positive pair: choose another sample with the same label.
            candidate_indices = self.label_to_indices[anchor_label].copy()
            if len(candidate_indices) == 1:
                # If only one sample is available, pair with itself.
                positive_idx = anchor_idx
            else:
                candidate_indices.remove(anchor_idx)
                positive_idx = np.random.choice(candidate_indices)
            pair_label = 1.0  # Similar pair
            pair_audio = self.audio_data[positive_idx]
            pair_landmarks = self.landmark_data[positive_idx]
        else:
            # Select a negative pair: choose a sample from a different label.
            possible_labels = list(self.label_to_indices.keys())
            possible_labels.remove(anchor_label)
            negative_label = np.random.choice(possible_labels)
            negative_idx = np.random.choice(self.label_to_indices[negative_label])
            pair_label = 0.0  # Dissimilar pair
            pair_audio = self.audio_data[negative_idx]
            pair_landmarks = self.landmark_data[negative_idx]
        
        # Optionally apply transforms.
        if self.transform:
            anchor_audio = self.transform(anchor_audio)
            anchor_landmarks = self.transform(anchor_landmarks)
            pair_audio = self.transform(pair_audio)
            pair_landmarks = self.transform(pair_landmarks)
        
        # Return a tuple: (audio1, landmarks1, audio2, landmarks2, label)
        return (anchor_audio, anchor_landmarks, pair_audio, pair_landmarks, torch.tensor(pair_label, dtype=torch.float32))
