import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, audio_dim=23, landmark_vector_dim=60, hidden_dim=128, num_classes=2):
        """
        Parameters:
          audio_dim: dimension of the audio feature vector (default: 23)
          landmark_vector_dim: dimension after pooling and flattening landmark data (default: 60)
          hidden_dim: number of hidden units in the classifier
          num_classes: number of output classes (2 for binary classification)
        """
        super(MLP, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(audio_dim + landmark_vector_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, audio, landmarks):
        """
        Parameters:
          audio: tensor of shape (batch, 23)
          landmarks: tensor of shape (batch, 250, 3, 20)
        """
        # Pool the landmark sequence along the time dimension (frame dimension)
        # Mean pooling over 250 frames --> shape becomes (batch, 3, 20)
        pooled_landmarks = landmarks.mean(dim=1)
        
        # Flatten the pooled landmarks to shape (batch, 3*20 = 60)
        pooled_landmarks = pooled_landmarks.view(pooled_landmarks.size(0), -1)
        fused_features = torch.cat([audio, pooled_landmarks], dim=1)
        
        out = self.classifier(fused_features)
        return out
