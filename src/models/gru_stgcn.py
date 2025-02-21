import torch
import torch.nn as nn


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


class GraphConv(nn.Module):
    def __init__(self, in_channels, out_channels, A, bias=True):
        """
        Graph convolution using a pre-defined adjacency matrix. with A registered as a buffer.
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
            # Using einsum to multiply along the node dimension.
            out = out + torch.einsum('nctv,vw->nctw', x[:, k], self.A[k])
        return out


class STGCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, A, kernel_size=9, stride=1, residual=True):
        """
        A single ST–GCN block that applies graph convolution and temporal convolution.
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
        """
        super(STGCNNet, self).__init__()
        A = adj.unsqueeze(0)  # shape: (1, num_nodes, num_nodes)
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


class MultiModalFusionNet(nn.Module):
    def __init__(self,
                 adjacency_matrix=torch.eye(20),
                 audio_input_size=23,
                 audio_hidden_size=128,
                 audio_num_layers=1,
                 bidirectional_audio=False,
                 stgcn_in_channels=3,
                 stgcn_hidden_channels=[64, 128],
                 fusion_type='self',   # Options: 'self', 'cross', or 'concat'
                 fusion_heads=4,
                 num_classes=2,
                 fusion_dropout=0.2):
        """
        Multi-modal network fusing GRU audio features with ST–GCN landmark features.
        Args:
            audio_input_size (int): Audio feature dimension.
            audio_hidden_size (int): GRU hidden size.
            audio_num_layers (int): Number of GRU layers.
            bidirectional_audio (bool): Whether to use bidirectional GRU.
            stgcn_in_channels (int): Landmark input channels (e.g., 3).
            stgcn_hidden_channels (list[int]): Hidden channels for ST–GCN blocks.
            fusion_type (str): Fusion method ('self', 'cross', or 'concat').
            fusion_heads (int): Number of heads for attention (if applicable).
            num_classes (int): Number of output classes.
        """
        super(MultiModalFusionNet, self).__init__()
        # Audio branch using GRU.
        self.audio_branch = AudioGRU(
            input_size=audio_input_size,
            hidden_size=audio_hidden_size,
            num_layers=audio_num_layers,
            bidirectional=bidirectional_audio
        )
        self.audio_rep_dim = audio_hidden_size * (2 if bidirectional_audio else 1)
        
        # Landmark branch using ST–GCN.
        self.landmark_branch = STGCNNet(
            adj=adjacency_matrix,
            num_nodes=20,          # 20 landmarks
            in_channels=stgcn_in_channels,
            hidden_channels=stgcn_hidden_channels,
            dropout=fusion_dropout
        )
        self.landmark_rep_dim = stgcn_hidden_channels[-1]
        
        self.fusion_type = fusion_type
        # Project both representations into a common fusion dimension.
        fusion_dim = max(self.audio_rep_dim, self.landmark_rep_dim)
        self.audio_proj = nn.Linear(self.audio_rep_dim, fusion_dim)
        self.landmark_proj = nn.Linear(self.landmark_rep_dim, fusion_dim)
        self.fusion_dim = fusion_dim
        
        if fusion_type == 'self':
            self.self_attn = nn.MultiheadAttention(embed_dim=fusion_dim, num_heads=fusion_heads, batch_first=True, dropout=fusion_dropout)
        elif fusion_type == 'cross':
            self.cross_attn_audio = nn.MultiheadAttention(embed_dim=fusion_dim, num_heads=fusion_heads, batch_first=True, dropout=fusion_dropout)
            self.cross_attn_landmark = nn.MultiheadAttention(embed_dim=fusion_dim, num_heads=fusion_heads, batch_first=True, dropout=fusion_dropout)
        elif fusion_type == 'concat':
            self.fc_fuse = nn.Sequential(
                nn.Linear(self.audio_rep_dim + self.landmark_rep_dim, fusion_dim),
                nn.ReLU(),
                nn.Dropout(fusion_dropout)
            )
        else:
            raise ValueError("fusion_type must be one of: 'self', 'cross', or 'concat'")
            
        self.fusion_dropout = nn.Dropout(fusion_dropout)
        self.classifier = nn.Linear(fusion_dim, num_classes)

    def forward(self, audio, landmarks):
        """
        Args:
            audio: Tensor of shape (N, T_audio, 23)
            landmarks: Tensor of shape (N, T_landmark, 3, 20)
        Returns:
            logits: Tensor of shape (N, num_classes)
        """
        N = audio.size(0)
        # Audio branch.
        audio_rep = self.audio_branch(audio)  # (N, audio_rep_dim)
        # Landmark branch: ST–GCN expects (N, channels, T, num_nodes)
        landmarks = landmarks.permute(0, 2, 1, 3)  # (N, 3, T_landmark, 20)
        landmark_rep = self.landmark_branch(landmarks)  # (N, landmark_rep_dim)
        
        if self.fusion_type in ['self', 'cross']:
            # Project to common fusion dimension.
            audio_proj = self.audio_proj(audio_rep)        # (N, fusion_dim)
            landmark_proj = self.landmark_proj(landmark_rep)   # (N, fusion_dim)
            if self.fusion_type == 'self':
                # Stack as tokens: (N, 2, fusion_dim) then self-attention.
                tokens = torch.stack([audio_proj, landmark_proj], dim=1)
                attn_out, _ = self.self_attn(tokens, tokens, tokens)
                fused = attn_out.mean(dim=1)  # (N, fusion_dim)
            else:  # 'cross'
                audio_q = audio_proj.unsqueeze(1)      # (N, 1, fusion_dim)
                landmark_q = landmark_proj.unsqueeze(1)  # (N, 1, fusion_dim)
                attn_audio, _ = self.cross_attn_audio(audio_q, landmark_q, landmark_q)
                attn_landmark, _ = self.cross_attn_landmark(landmark_q, audio_q, audio_q)
                fused = (attn_audio.squeeze(1) + attn_landmark.squeeze(1)) / 2
        elif self.fusion_type == 'concat':
            fused = self.fc_fuse(torch.cat([audio_rep, landmark_rep], dim=1))
        else:
            raise ValueError("Invalid fusion_type")
        
        fused = self.fusion_dropout(fused)
        logits = self.classifier(fused)  # (N, num_classes)
        return logits
