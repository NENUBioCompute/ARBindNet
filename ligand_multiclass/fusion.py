import torch
import torch.nn as nn
from torch_scatter import scatter_max

class CrossLevelFusion(nn.Module):
    """
    将原子池化特征与残基本征特征进行交叉注意力融合。
    """
    def __init__(self, hidden_dim, num_heads=4, dropout=0.3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.cross_attn_atom_to_res = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.cross_attn_res_to_atom = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.norm_out = nn.LayerNorm(hidden_dim)

    def forward(self, h_atom_pooled, h_res, batch_idx):
        # 构造跨图掩码：同一蛋白质内允许注意，不同蛋白质间屏蔽
        same_graph_mask = (batch_idx.unsqueeze(0) == batch_idx.unsqueeze(1))
        attn_mask = torch.zeros_like(same_graph_mask, dtype=torch.float)
        attn_mask.masked_fill_(~same_graph_mask, float('-inf'))

        # 添加 batch 维度 (1, N_total, hidden_dim)
        q_res = h_res.unsqueeze(0)
        kv_atom = h_atom_pooled.unsqueeze(0)

        # 交叉注意力
        attn_out1, _ = self.cross_attn_atom_to_res(q_res, kv_atom, kv_atom, attn_mask=attn_mask)
        h_res1 = self.norm1(q_res + attn_out1)

        attn_out2, _ = self.cross_attn_res_to_atom(kv_atom, h_res1, h_res1, attn_mask=attn_mask)
        h_atom2 = self.norm2(kv_atom + attn_out2)

        # 拼接、融合、全局池化
        h_cat = torch.cat([h_res1, h_atom2], dim=-1).squeeze(0)
        h_fused = self.norm_out(self.ffn(h_cat))
        # h_graph = self._hybrid_pool_per_graph(h_fused, batch_idx)
        h_graph = self._avg_pool_per_graph(h_fused, batch_idx)

        return h_fused, h_graph

    def _avg_pool_per_graph(self, x, batch_idx):
        num_graphs = batch_idx.max().item() + 1
        sum_pooled = torch.zeros(num_graphs, x.size(1), device=x.device)
        sum_pooled.index_add_(0, batch_idx, x)
        count = torch.bincount(batch_idx, minlength=num_graphs).float().clamp(min=1).unsqueeze(-1)
        return sum_pooled / count

    # def _hybrid_pool_per_graph(self, x, batch_idx):
    #     # 1. 平均池化
    #     num_graphs = batch_idx.max().item() + 1
    #     sum_pooled = torch.zeros(num_graphs, x.size(1), device=x.device)
    #     sum_pooled.index_add_(0, batch_idx, x)
    #     count = torch.bincount(batch_idx, minlength=num_graphs).float().clamp(min=1).unsqueeze(-1)
    #     h_avg = sum_pooled / count
    #     # 2. 最大池化 (使用 torch_scatter 的 scatter_max 更快)
    #     h_max, _ = scatter_max(x, batch_idx, dim=0)
    #     # 3. 拼接
    #     h_combined = torch.cat([h_avg, h_max], dim=-1)  # 维度变为 hidden_dim * 2
    #     return h_combined