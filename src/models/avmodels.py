import torch
import torch.nn as nn

class AudioLSTM(nn.Module):
    def __init__(self, 
                 audio_input_size=23, 
                 lstm_hidden_size=64, 
                 lstm_num_layers=1, 
                 dropout=0.8,
                 num_classes=2):
        """
        Args:
            audio_input_size (int): Number of features for audio input.
            lstm_hidden_size (int): Hidden state dimension for the LSTM.
            lstm_num_layers (int): Number of LSTM layers.
            dropout (float): Dropout probability.
            num_classes (int): Number of output classes.
        """
        super(AudioLSTM, self).__init__()
        
        # Audio branch LSTM
        self.audio_lstm = nn.LSTM(
            input_size=audio_input_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=dropout if lstm_num_layers > 1 else 0
        )
        
        # Final Classification Layer
        self.fc = nn.Linear(lstm_hidden_size, num_classes)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, audio):
        """
        Args:
            audio (torch.Tensor): Audio input of shape (batch, seq_len, 23).
            
        Returns:
            logits (torch.Tensor): Logits of shape (batch, num_classes).
        """
        # Process audio input through its LSTM.
        audio_out, _ = self.audio_lstm(audio)  # (batch, seq_len, lstm_hidden_size)
        
        # Mean pooling over time.
        audio_rep = torch.mean(audio_out, dim=1)  # (batch, lstm_hidden_size)
        audio_rep = self.dropout(audio_rep)
        
        logits = self.fc(audio_rep)  # (batch, num_classes)
        return logits


class AudioLSTMAttn(nn.Module):
    def __init__(self, 
            audio_input_size=23, 
            lstm_hidden_size=64, 
            lstm_num_layers=1, 
            dropout=0.8,
            num_classes=2,
            num_heads=4):
        super(AudioLSTMAttn, self).__init__()
        
        # Audio branch LSTM
        self.audio_lstm = nn.LSTM(
            input_size=audio_input_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=dropout if lstm_num_layers > 1 else 0
        )
        
        # Multiheaded attention layer for self attention.
        # nn.MultiheadAttention expects input shape (seq_len, batch, embed_dim).
        self.attention = nn.MultiheadAttention(
            embed_dim=lstm_hidden_size, 
            num_heads=num_heads, 
            dropout=dropout
        )
        
        # Final Classification Layer
        self.fc = nn.Linear(lstm_hidden_size, num_classes)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, audio):
        """
        Args:
            audio (torch.Tensor): Audio input of shape (batch, seq_len, audio_input_size).
            
        Returns:
            logits (torch.Tensor): Logits of shape (batch, num_classes).
        """
        # Process audio input through its LSTM.
        lstm_out, _ = self.audio_lstm(audio)  # (batch, seq_len, lstm_hidden_size)
        
        # Prepare LSTM output for multihead attention: (seq_len, batch, lstm_hidden_size)
        lstm_out_t = lstm_out.transpose(0, 1)
        
        # Self-attention using multihead attention.
        attn_out, _ = self.attention(lstm_out_t, lstm_out_t, lstm_out_t)
        attn_out = attn_out.transpose(0, 1)  # (batch, seq_len, lstm_hidden_size)
        
        # Mean pooling over time.
        audio_rep = torch.mean(attn_out, dim=1)  # (batch, lstm_hidden_size)
        audio_rep = self.dropout(audio_rep)
        
        logits = self.fc(audio_rep)  # (batch, num_classes)
        return logits


class AudioGRU(nn.Module):
    def __init__(self, input_size=23, hidden_size=128, num_layers=1, bidirectional=False, dropout=0.2):
        """
        GRU-based branch for audio modality.
        Args:
            input_size (int): Number of audio features per timestep.
            hidden_size (int): Hidden size of the GRU.
            num_layers (int): Number of GRU layers.
            bidirectional (bool): Whether to use bidirectional GRU.
            dropout (float): Dropout probability (applied between layers if num_layers > 1).
        """
        super(AudioGRU, self).__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        self.hidden_size = hidden_size
        self.num_directions = 2 if bidirectional else 1
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: (N, T_audio, input_size)
        out, _ = self.gru(x)  # shape: (N, T_audio, hidden_size * num_directions)
        # Temporal average pooling to summarize the sequence.
        out = torch.mean(out, dim=1)  # (N, hidden_size * num_directions)
        out = self.dropout(out)
        return out


class AudioGRUNet(nn.Module):
    def __init__(self, 
                 input_size=23, 
                 hidden_size=128, 
                 num_layers=1, 
                 bidirectional=False, 
                 gru_dropout=0.2, 
                 num_heads=4, 
                 classifier_dropout=0.2,
                 num_classes=2):
        """
        Audio network built on top of the GRU branch.
        A self-attention module processes the encoded audio feature before classification.
        Args:
            input_size (int): Audio feature dimension.
            hidden_size (int): GRU hidden size.
            num_layers (int): GRU number of layers.
            bidirectional (bool): Whether the GRU is bidirectional.
            gru_dropout (float): Dropout for GRU.
            num_heads (int): Number of attention heads.
            classifier_dropout (float): Dropout before classification.
            num_classes (int): Number of output classes.
        """
        super(AudioGRUNet, self).__init__()
        self.audio_branch = AudioGRU(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            bidirectional=bidirectional, 
            dropout=gru_dropout
        )
        self.rep_dim = hidden_size * (2 if bidirectional else 1)
        # Self-attention over a token sequence of length 1.
        self.self_attn = nn.MultiheadAttention(
            embed_dim=self.rep_dim, 
            num_heads=num_heads, 
            batch_first=True, 
            dropout=classifier_dropout
        )
        self.classifier = nn.Linear(self.rep_dim, num_classes)
        self.classifier_dropout = nn.Dropout(classifier_dropout)

    def forward(self, audio):
        """
        Args:
            audio (torch.Tensor): Audio input of shape (N, T_audio, input_size).
        Returns:
            logits (torch.Tensor): Logits of shape (N, num_classes).
        """
        rep = self.audio_branch(audio)  # (N, rep_dim)
        # Expand to sequence with length 1 for self-attention.
        rep = rep.unsqueeze(1)  # (N, 1, rep_dim)
        attn_out, _ = self.self_attn(rep, rep, rep)
        rep = attn_out.squeeze(1)  # (N, rep_dim)
        rep = self.classifier_dropout(rep)
        logits = self.classifier(rep)
        return logits


class GraphConv(nn.Module):
    def __init__(self, in_channels, out_channels, A, bias=True):
        """
        Graph convolution using a pre-defined adjacency matrix.
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            A (torch.Tensor): Adjacency matrix of shape (K, V, V), where K is number of partitions.
            bias (bool): Whether to include bias.
        """
        super(GraphConv, self).__init__()
        # Register A as a buffer so it moves with the module.
        self.register_buffer('A', A)
        self.K = A.size(0)
        # 1x1 convolution to produce K*out_channels channels.
        self.conv = nn.Conv2d(in_channels, out_channels * self.K, kernel_size=1, bias=bias)

    def forward(self, x):
        # x shape: (N, in_channels, T, V)
        N, C, T, V = x.size()
        x = self.conv(x)  # (N, out_channels*K, T, V)
        # Reshape to (N, K, out_channels, T, V)
        x = x.view(N, self.K, -1, T, V)
        out = 0
        # Multiply each partition with the corresponding adjacency matrix.
        for k in range(self.K):
            out = out + torch.einsum('nctv,vw->nctw', x[:, k], self.A[k])
        return out


# A single ST–GCN block that applies graph convolution and temporal convolution.
class STGCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, A, kernel_size=9, stride=1, residual=True):
        """
        ST–GCN block.
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            A (torch.Tensor): Adjacency matrix (shape: (K, V, V)).
            kernel_size (int): Temporal convolution kernel size.
            stride (int): Temporal stride.
            residual (bool): Whether to use a residual connection.
        """
        super(STGCNBlock, self).__init__()
        self.gcn = GraphConv(in_channels, out_channels, A)
        # Temporal convolution: kernel (kernel_size, 1)
        padding = ((kernel_size - 1) // 2, 0)
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=(kernel_size, 1), stride=(stride, 1), padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(0.5, inplace=True)
        )
        if not residual:
            self.residual = lambda x: 0
        elif in_channels == out_channels and stride == 1:
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels)
            )
        self.relu = nn.ReLU()

    def forward(self, x):
        res = self.residual(x)
        x = self.gcn(x)
        x = self.tcn(x)
        x = x + res
        return self.relu(x)


class STGCNNet(nn.Module):
    def __init__(self, adj, in_channels=3, hidden_channels=[64, 128], dropout=0.2):
        """
        ST–GCN network for landmarks.
        Args:
            num_nodes (int): Number of graph nodes (e.g., 20 landmarks).
            in_channels (int): Number of input channels (e.g., 3 coordinates).
            hidden_channels (list[int]): Output channels for each ST–GCN block.
            dropout (float): Dropout probability.
        """
        super(STGCNNet, self).__init__()
        # Use an identity matrix as the adjacency matrix.
        A = adj.unsqueeze(0)    # shape: (1, num_nodes, num_nodes)
        self.block1 = STGCNBlock(in_channels, hidden_channels[0], A)
        self.block2 = STGCNBlock(hidden_channels[0], hidden_channels[1], A)
        # Global pooling over temporal and spatial dimensions.
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x expected: (N, in_channels, T, num_nodes)
        x = self.block1(x)
        x = self.block2(x)
        x = self.pool(x)  # (N, hidden_channels[-1], 1, 1)
        x = x.view(x.size(0), -1)  # (N, hidden_channels[-1])
        x = self.dropout(x)
        return x


class LandmarkSTGCNNet(nn.Module):
    def __init__(self,
                 adjacency_matrix=torch.eye(20),
                 num_nodes=20, 
                 in_channels=3, 
                 hidden_channels=[64, 128], 
                 stgcn_dropout=0.2, 
                 num_heads=4, 
                 classifier_dropout=0.2,
                 num_classes=2):
        """
        Landmark network built on top of the ST–GCN branch.
        A self-attention module processes the landmark feature before classification.
        Args:
            num_nodes (int): Number of landmarks.
            in_channels (int): Input channels (e.g., 3 for coordinates).
            hidden_channels (list[int]): Hidden channels for ST–GCN blocks.
            stgcn_dropout (float): Dropout for ST–GCN.
            num_heads (int): Number of attention heads.
            classifier_dropout (float): Dropout before classification.
            num_classes (int): Number of output classes.
        """
        super(LandmarkSTGCNNet, self).__init__()
        self.landmark_branch = STGCNNet(
            adjacency_matrix,
            num_nodes=num_nodes,
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            dropout=stgcn_dropout
        )
        self.rep_dim = hidden_channels[-1]
        # Self-attention over a token sequence of length 1.
        self.self_attn = nn.MultiheadAttention(
            embed_dim=self.rep_dim, 
            num_heads=num_heads, 
            batch_first=True, 
            dropout=classifier_dropout
        )
        self.classifier = nn.Linear(self.rep_dim, num_classes)
        self.classifier_dropout = nn.Dropout(classifier_dropout)

    def forward(self, landmarks):
        """
        Args:
            landmarks (torch.Tensor): Landmark input of shape (N, T_landmark, in_channels, num_nodes).
        Returns:
            logits (torch.Tensor): Logits of shape (N, num_classes).
        """
        # ST–GCN expects input shape (N, in_channels, T, num_nodes)
        landmarks = landmarks.permute(0, 2, 1, 3)  # (N, in_channels, T, num_nodes)
        rep = self.landmark_branch(landmarks)  # (N, rep_dim)
        # Expand to sequence with length 1 for self-attention.
        rep = rep.unsqueeze(1)  # (N, 1, rep_dim)
        attn_out, _ = self.self_attn(rep, rep, rep)
        rep = attn_out.squeeze(1)  # (N, rep_dim)
        rep = self.classifier_dropout(rep)
        logits = self.classifier(rep)
        return logits


class LandmarkLSTM(nn.Module):
    def __init__(self, 
                 landmark_input_size=60,  # 3 x 20 = 60 after reshaping
                 lstm_hidden_size=64, 
                 lstm_num_layers=1, 
                 dropout=0.8,
                 num_classes=2):
        """
        Args:
            landmark_input_size (int): Number of features for landmark input (after reshaping).
            lstm_hidden_size (int): Hidden state dimension for landmark LSTM.
            lstm_num_layers (int): Number of LSTM layers.
            dropout (float): Dropout probability.
            num_classes (int): Number of output classes.
        """
        super(LandmarkLSTM, self).__init__()
        
        # ----- Landmark branch LSTM -----
        # Landmarks: (batch, seq_len_landmark, 3, 20) -> reshape to (batch, seq_len_landmark, 60)
        self.landmark_lstm = nn.LSTM(
            input_size=landmark_input_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=dropout if lstm_num_layers > 1 else 0
        )
        
        # ----- Final Classification Layer -----
        self.fc = nn.Linear(lstm_hidden_size, num_classes)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, landmarks):
        """
        Args:
            landmarks (torch.Tensor): Landmark input of shape (batch, seq_len_landmark, 3, 20).
            
        Returns:
            logits (torch.Tensor): Logits of shape (batch, num_classes).
        """
        batch_size = landmarks.size(0)
        # Reshape landmarks from (batch, seq_len_landmark, 3, 20) to (batch, seq_len_landmark, 60)
        landmarks = landmarks.view(batch_size, landmarks.size(1), -1)
        landmark_out, _ = self.landmark_lstm(landmarks)  # (batch, seq_len_landmark, lstm_hidden_size)
        landmark_rep = torch.mean(landmark_out, dim=1)  # (batch, lstm_hidden_size)
        landmark_rep = self.dropout(landmark_rep)
        logits = self.fc(landmark_rep)  # (batch, num_classes)
        return logits
