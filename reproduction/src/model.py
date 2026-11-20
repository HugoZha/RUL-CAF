"""CAS-Fuse hybrid backbone (TCN + BiGRU + Transformer), no adversarial branch.

Faithful extraction of ``PureRULModel`` from the experiment script that
produced every reported number.  The only addition is an optional
``return_attn`` flag used by the R5 attention-entropy tripwire; it does not
change the computation or the random-number consumption of a forward pass.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, dilation=1):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size,
                              padding=self.pad, dilation=dilation)

    def forward(self, x):
        x = self.conv(x)
        if self.pad > 0:
            x = x[:, :, :-self.pad]
        return x


class TemporalBlock(nn.Module):
    def __init__(self, ch, kernel_size=3, dilation=1, dropout=0.1):
        super().__init__()
        self.causal = CausalConv1d(ch, ch, kernel_size, dilation)
        self.norm = nn.LayerNorm(ch)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.res_proj = nn.Conv1d(ch, ch, 1) if dilation > 1 else nn.Identity()

    def forward(self, x):
        h = self.causal(x)
        h = h.transpose(1, 2)
        h = self.norm(h)
        h = h.transpose(1, 2)
        h = self.act(h)
        h = self.drop(h)
        res = self.res_proj(x)
        if res.shape[-1] > h.shape[-1]:
            res = res[..., :h.shape[-1]]
        elif res.shape[-1] < h.shape[-1]:
            res = F.pad(res, (h.shape[-1] - res.shape[-1], 0))
        return h + res


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float)
                        * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads=4, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.scale = math.sqrt(self.d_k)
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        B, L, _ = x.size()
        Q = self.W_q(x).view(B, L, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(B, L, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(B, L, self.num_heads, self.d_k).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        causal = torch.triu(torch.ones(L, L, device=x.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(causal, float("-inf"))
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        attn = self.drop(attn)
        context = torch.matmul(attn, V)
        context = context.transpose(1, 2).contiguous().view(B, L, self.num_heads * self.d_k)
        return self.W_o(context), attn


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=512, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model), nn.Dropout(dropout))

    def forward(self, x):
        return self.net(x)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads=4, d_ff=512, dropout=0.1):
        super().__init__()
        self.attn_norm = nn.LayerNorm(d_model)
        self.attn_sublayer = MultiHeadAttention(d_model, num_heads, dropout)
        self.drop1 = nn.Dropout(dropout)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn_sublayer = FeedForward(d_model, d_ff, dropout)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x):
        h = self.attn_norm(x)
        h, attn = self.attn_sublayer(h)
        x = x + self.drop1(h)
        h = self.ffn_norm(x)
        h = self.ffn_sublayer(h)
        x = x + self.drop2(h)
        return x, attn


class SensorDropout(nn.Module):
    def __init__(self, p=0.1):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training:
            return x
        return x * (torch.rand(x.shape, device=x.device) > self.p).float()


class PureRULModel(nn.Module):
    """TCN -> BiGRU -> Transformer encoder -> dual (avg+max) pooling head.

    Difference augmentation (x concatenated with first-difference) and
    sensor-channel dropout are always on, per rules R3-R4 of Sec. 5.1.
    """

    def __init__(self, n_features, hidden_dim=128, num_heads=4, num_layers=2,
                 dropout=0.3, sensor_dropout_p=0.1, n_encoder_layers=1):
        super().__init__()
        total_features = n_features * 2  # x + diff
        self.input_proj = nn.Sequential(
            nn.Conv1d(total_features, hidden_dim, kernel_size=1),
            nn.GELU(), nn.Dropout(dropout))
        self.pos_encoder = PositionalEncoding(hidden_dim, max_len=128, dropout=dropout)
        self.sensor_dropout = SensorDropout(p=sensor_dropout_p)
        dilations = [1, 2, 4, 8, 16]
        self.tcn_layers = nn.ModuleList([
            TemporalBlock(hidden_dim, kernel_size=3, dilation=d, dropout=dropout)
            for d in dilations])
        self.gru = nn.GRU(input_size=hidden_dim, hidden_size=hidden_dim,
                          num_layers=num_layers, batch_first=True, bidirectional=True,
                          dropout=dropout if num_layers > 1 else 0)
        self.gru_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model=hidden_dim, num_heads=num_heads,
                                    d_ff=hidden_dim * 4, dropout=dropout)
            for _ in range(n_encoder_layers)])
        self.encoder_norm = nn.LayerNorm(hidden_dim)
        self.rul_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.LayerNorm(hidden_dim),
            nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1))
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, return_attn=False):
        diff = torch.diff(x, dim=1, prepend=x[:, :1, :])
        x_aug = torch.cat([x, diff], dim=2)
        h = x_aug.transpose(1, 2)
        h = self.input_proj(h)
        h = h.transpose(1, 2)
        h = self.pos_encoder(h)
        h = self.sensor_dropout(h)
        h_cnn = h.transpose(1, 2)
        for layer in self.tcn_layers:
            h_cnn = layer(h_cnn)
        h = h + h_cnn.transpose(1, 2)
        h_gru, _ = self.gru(h)
        h = self.gru_proj(h_gru) + h
        attn_last = None
        for layer in self.encoder_layers:
            h, attn_last = layer(h)
        h = self.encoder_norm(h)
        h_avg = h.mean(dim=1)
        h_max, _ = h.max(dim=1)
        h_pool = torch.cat([h_avg, h_max], dim=1)
        rul_pred = self.rul_head(h_pool)
        if return_attn:
            return rul_pred, attn_last
        return rul_pred


def build_model(n_features, cfg: dict, fixed: dict) -> PureRULModel:
    """Build a model from ``configs.BEST_CONFIG`` / ``configs.FIXED_CONFIG``."""
    return PureRULModel(
        n_features=n_features, hidden_dim=fixed["hidden_dim"],
        num_heads=cfg["num_heads"], num_layers=fixed["gru_num_layers"],
        dropout=fixed["dropout"], sensor_dropout_p=fixed["sensor_dropout_p"],
        n_encoder_layers=cfg["n_encoder_layers"])
