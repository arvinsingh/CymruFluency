import torch
import torch.nn as nn


class MultiModalLSTM(nn.Module):
    def __init__(self, 
                 audio_input_size=23, 
                 landmark_input_size=60,  # 3 x 20 = 60 after reshaping
                 lstm_hidden_size=64, 
                 lstm_num_layers=1, 
                 attn_heads=2, 
                 dropout=0.8,
                 fusion_dropout=0.8,
                 num_classes=2,
                 fusion_type="self"):
                 # fusion_type options: "self", "cross", "concat"
        """
        Args:
            audio_input_size (int): Number of features for audio input.
            landmark_input_size (int): Number of features for landmark input (after reshaping).
            lstm_hidden_size (int): Hidden state dimension for both LSTMs.
            lstm_num_layers (int): Number of LSTM layers.
            attn_heads (int): Number of heads for attention modules.
            dropout (float): Dropout probability.
            num_classes (int): Number of output classes.
            fusion_type (str): Fusion option; "self" applies self-attention on stacked tokens,
                               "cross" uses cross attention (audio as query, landmark as key/value),
                               and "concat" simply concatenates the two representations.
        """
        super(MultiModalLSTM, self).__init__()
        self.fusion_type = fusion_type.lower()

        # ----- Audio branch LSTM -----
        self.audio_lstm = nn.LSTM(
            input_size=audio_input_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=dropout if lstm_num_layers > 1 else 0
        )
        
        # ----- Landmark branch LSTM -----
        # Landmarks: (batch, frames, 3, 20) -> reshape to (batch, frames, 60)
        self.landmark_lstm = nn.LSTM(
            input_size=landmark_input_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=dropout if lstm_num_layers > 1 else 0
        )

        # ----- Fusion Layer -----
        if self.fusion_type in ["self", "cross"]:
            # For both self and cross attention, use multihead attention.
            # nn.MultiheadAttention expects (seq_len, batch, embed_dim)
            self.attn = nn.MultiheadAttention(
                embed_dim=lstm_hidden_size,
                num_heads=attn_heads,
                dropout=fusion_dropout
            )
            # Final classification: input dimension remains lstm_hidden_size.
        elif self.fusion_type == "concat":
            # No attention module needed.
            # After concatenation, the dimension is 2 * lstm_hidden_size.
            self.fc_fuse = nn.Sequential(
                nn.Linear(2 * lstm_hidden_size, lstm_hidden_size),
                nn.ReLU(),
                nn.Dropout(fusion_dropout)
            )
        else:
            raise ValueError("Invalid fusion_type. Expected one of: 'self', 'cross', 'concat'.")
        
        self.fusion_dropout = nn.Dropout(fusion_dropout)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(lstm_hidden_size, num_classes)

    def forward(self, audio, landmarks):
        """
        Args:
            audio (torch.Tensor): Audio input of shape (batch, seq_len_audio, 23).
            landmarks (torch.Tensor): Landmark input of shape (batch, seq_len_landmark, 3, 20).
            
        Returns:
            logits (torch.Tensor): Logits of shape (batch, num_classes).
        """
        batch_size = audio.size(0)
        
        # ----- Audio branch -----
        audio_out, _ = self.audio_lstm(audio)  # (batch, seq_len_audio, lstm_hidden_size)
        audio_rep = torch.mean(audio_out, dim=1)  # (batch, lstm_hidden_size)
        
        # ----- Landmark branch -----
        landmarks = landmarks.view(batch_size, landmarks.size(1), -1)
        landmark_out, _ = self.landmark_lstm(landmarks)  # (batch, seq_len_landmark, lstm_hidden_size)
        landmark_rep = torch.mean(landmark_out, dim=1)  # (batch, lstm_hidden_size)
        
        # ----- Fusion -----
            # ----- For self fusion type -----
        if self.fusion_type == "self":
            # Stack the representations to create a sequence of 2 tokens.
            tokens = torch.stack([audio_rep, landmark_rep], dim=1)  # (batch, 2, lstm_hidden_size)
            # Transpose to shape (seq_len, batch, embed_dim)
            tokens = tokens.transpose(0, 1)  # (2, batch, lstm_hidden_size)
            attn_output, _ = self.attn(tokens, tokens, tokens)
            fused_rep = torch.mean(attn_output, dim=0)  # (batch, lstm_hidden_size)
        
        # ----- For cross fusion type: audio attends to landmarks AND landmarks attend to audio -----
        elif self.fusion_type == "cross":
            # Audio attends to landmarks.
            query_audio = audio_rep.unsqueeze(0)             # (1, batch, lstm_hidden_size)
            key_value_landmark = landmark_rep.unsqueeze(0)     # (1, batch, lstm_hidden_size)
            attn_audio, _ = self.attn(query_audio, key_value_landmark, key_value_landmark)
            
            # Landmarks attend to audio.
            query_landmark = landmark_rep.unsqueeze(0)         # (1, batch, lstm_hidden_size)
            key_value_audio = audio_rep.unsqueeze(0)           # (1, batch, lstm_hidden_size)
            attn_landmark, _ = self.attn(query_landmark, key_value_audio, key_value_audio)
            
            # Combine both representations (averaging).
            fused_rep = (attn_audio.squeeze(0) + attn_landmark.squeeze(0)) / 2
        
        elif self.fusion_type == "concat":
            fused_rep = self.fc_fuse(torch.cat([audio_rep, landmark_rep], dim=-1))  # (batch, 2 * lstm_hidden_size)
        
        fused_rep = self.fusion_dropout(fused_rep)
        logits = self.classifier(fused_rep)  # (batch, num_classes)

        return logits
