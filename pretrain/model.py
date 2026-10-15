# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
import dgl.nn.functional as dglF
from loss import random_mask_atom
from torch_scatter import scatter_softmax, scatter_sum

# =========================================================
# 1. EGNN Encoder (Atom Graph) with MAE mask support
# =========================================================
class EGNNLayer(nn.Module):
    """带注意力/无注意力选择的等变图神经网络单层"""

    def __init__(self, hidden_dim, edge_dim, num_heads=4, use_attention=True, dropout=0.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.use_attention = use_attention
        self.dropout = dropout

        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + 1 + edge_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_heads if use_attention else 1)
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.norm_h = nn.LayerNorm(hidden_dim)

        if use_attention:
            self.attn_mlp = nn.Linear(hidden_dim, num_heads)

    def forward(self, g, h, x, u=None):
        src, dst = g.edges()

        hi, hj = h[src], h[dst]
        xi, xj = x[src], x[dst]

        dist_sq = ((xi - xj) ** 2).sum(dim=-1, keepdim=True)
        edge_feat = g.edata['edge_attr']
        m_ij = self.edge_mlp(torch.cat([hi, hj, dist_sq, edge_feat], dim=-1))

        if not self.use_attention:
            g.edata['m'] = m_ij
            g.update_all(fn.copy_e('m', 'm'), fn.sum('m', 'm_sum'))

            coord_gate = self.coord_mlp(m_ij)
            diff = xi - xj
            g.edata['coord_msg'] = diff * coord_gate
            g.update_all(fn.copy_e('coord_msg', 'c'), fn.sum('c', 'c_sum'))

            deg = g.in_degrees().float().clamp(min=1).unsqueeze(-1)
            x_new = x + g.ndata['c_sum'] / deg

            h_new = self.node_mlp(torch.cat([h, g.ndata['m_sum']], dim=-1))
            # h_new = self.norm_h(h_new)

        else:
            M = self.num_heads
            d = self.hidden_dim // M

            coord_gate = self.coord_mlp(m_ij)      # (E, M)
            attn_logits = self.attn_mlp(m_ij)      # (E, M)
            m_ij_heads = m_ij.view(-1, M, d)       # (E, M, d)

            g.edata['attn_logits'] = attn_logits
            g.edata['m_heads'] = m_ij_heads
            g.edata['coord_gate'] = coord_gate
            g.edata['diff'] = xi - xj

            # 注意力归一化
            a_norm = dglF.edge_softmax(g, g.edata['attn_logits'])
            g.edata['a_norm'] = a_norm

            def message_func(edges):
                return {
                    'm_heads': edges.data['m_heads'],
                    'coord_gate': edges.data['coord_gate'],
                    'diff': edges.data['diff'],
                    'a_norm': edges.data['a_norm']
                }

            def reduce_func(nodes):
                m_heads = nodes.mailbox['m_heads']
                c_gate = nodes.mailbox['coord_gate']
                diff = nodes.mailbox['diff']
                a = nodes.mailbox['a_norm']

                # 坐标更新 (加权平均)
                diff_exp = diff.unsqueeze(2)          # (E,1,3)
                c_gate_exp = c_gate.unsqueeze(-1)     # (E,M,1)
                a_exp = a.unsqueeze(-1)               # (E,M,1)
                coord_msg = diff_exp * c_gate_exp * a_exp
                coord_agg = coord_msg.sum(dim=1)      # 对边求和 -> (N, M, 3)
                coord_total = coord_agg.sum(dim=1)    # 边维度求和 -> (N,3)
                # coord_total = coord_total / nodes.mailbox['diff'].shape[1]
                # deg = nodes.mailbox['diff'].shape[1]  # 每节点的入度，形状 (N,)
                # deg = deg.float().clamp(min=1).unsqueeze(-1)  # (N, 1) 防止除零
                # coord_total = coord_total / deg

                # 消息聚合
                node_msg = m_heads * a.unsqueeze(-1)  # (N, deg, M, d)
                node_agg = node_msg.sum(dim=1)        # (N, M, d)
                h_msg_total = node_agg.reshape(node_agg.size(0), -1)  # (N, M*d)

                return {'coord_sum': coord_total, 'h_msg': h_msg_total}

            g.update_all(message_func, reduce_func)
            deg = g.in_degrees().float().clamp(min=1).to(x.device).unsqueeze(-1)
            x_new = x + g.ndata['coord_sum']
            h_new = self.node_mlp(torch.cat([h, g.ndata['h_msg']], dim=-1))
            # h_new = self.norm_h(h_new)

        return h_new, x_new


class EGNNEncoder(nn.Module):
    """EGNN编码器，支持随机掩码 (MAE)"""

    def __init__(self, node_dim, edge_dim, hidden_dim, num_layers,
                 num_heads=4, use_attention=True, coord_noise_std=0.3,dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(node_dim, hidden_dim)
        self.norm_in = nn.LayerNorm(hidden_dim)
        self.layers = nn.ModuleList([
            EGNNLayer(hidden_dim, edge_dim, num_heads, use_attention, dropout=dropout)
            for _ in range(num_layers)
        ])
        # 可学习的掩码标记，用于替换被遮蔽原子的标量特征
        self.mask_token = nn.Parameter(torch.zeros(1, hidden_dim))
        # 给被遮蔽原子坐标添加的高斯噪声标准差
        self.coord_noise_std = coord_noise_std

    def forward(self, g, mask=None):
        h = self.proj(g.ndata['feat'])
        h = self.norm_in(h)
        x = g.ndata['x'].clone()   # 避免直接修改原图数据

        if mask is not None:
            # 被遮蔽原子的标量特征替换为 mask_token
            h[mask] = self.mask_token.expand(mask.sum(), -1)
            # 坐标添加噪声，防止坐标泄露
            noise = torch.randn_like(x[mask]) * self.coord_noise_std
            x[mask] = x[mask] + noise
            # 也可以完全替换为噪声：x[mask] = noise

        for layer in self.layers:
            h, x = layer(g, h, x)

        return h, x


# =========================================================
# 2. GVP Encoder (Residue Graph) — unchanged from previous
# =========================================================
class GVP(nn.Module):
    """几何向量感知器模块（Geometric Vector Perceptron）"""

    def __init__(self, dim_s_in, dim_v_in, dim_s_out, dim_v_out, dropout=0.1):
        super().__init__()
        self.dim_v_in = dim_v_in
        self.dim_v_out = dim_v_out

        self.W_s = nn.Sequential(
            nn.Linear(dim_s_in + dim_v_in, dim_s_out),
            nn.SiLU()
        )
        self.W_v = nn.Linear(dim_v_in, dim_v_out, bias=False)

        self.gate = nn.Sequential(
            nn.Linear(dim_s_in + dim_v_in, dim_v_out),
            nn.Sigmoid()
        )
        self.s_from_v = nn.Linear(dim_v_in, dim_v_in)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, s, V):
        v_norm = torch.sqrt(torch.sum(V**2, dim=-1) + 1e-8)
        s_v = self.s_from_v(v_norm)
        s_cat = torch.cat([s, s_v], dim=-1)

        s_out = self.W_s(s_cat)
        V_proj = self.W_v(V.transpose(-1, -2)).transpose(-1, -2)
        gate = self.gate(s_cat).unsqueeze(-1)
        V_out = V_proj * gate

        return self.dropout(s_out), V_out


class GVPConvLayer(nn.Module):
    """GVP图卷积层"""

    def __init__(self, hidden_s_dim, hidden_v_dim, dropout=0.1):
        super().__init__()
        self.message_gvp = GVP(
            dim_s_in=3 * hidden_s_dim,
            dim_v_in=2 * hidden_v_dim,
            dim_s_out=hidden_s_dim,
            dim_v_out=hidden_v_dim
        )
        self.update_gvp = GVP(
            dim_s_in=hidden_s_dim,
            dim_v_in=hidden_v_dim,
            dim_s_out=hidden_s_dim,
            dim_v_out=hidden_v_dim
        )
        self.norm_s = nn.LayerNorm(hidden_s_dim)
        self.dropout = nn.Dropout(dropout)

    def message_func(self, edges):
        s_input = torch.cat([edges.src["s"], edges.dst["s"], edges.data["s"]], dim=-1)
        V_input = torch.cat([edges.src["V"], edges.data["V"]], dim=1)
        s_msg, V_msg = self.message_gvp(s_input, V_input)
        return {"s_msg": s_msg, "V_msg": V_msg}

    # def reduce_func(self, nodes):
    #     s_agg = nodes.mailbox["s_msg"].sum(dim=1)
    #     V_agg = nodes.mailbox["V_msg"].sum(dim=1)
    #     return {"s_agg": s_agg, "V_agg": V_agg}
    def reduce_func(self, nodes):
        s_msg = nodes.mailbox["s_msg"]
        V_msg = nodes.mailbox["V_msg"]

        deg = s_msg.shape[1]
        norm = s_msg.new_tensor(deg).sqrt()

        s_agg = s_msg.sum(dim=1) / norm
        V_agg = V_msg.sum(dim=1) / norm

        return {"s_agg": s_agg, "V_agg": V_agg}

    def forward(self, g):
        g.update_all(self.message_func, self.reduce_func)

        s_agg = g.ndata.pop("s_agg")
        V_agg = g.ndata.pop("V_agg")

        s = g.ndata["s"]
        V = g.ndata["V"]

        s = s + self.dropout(s_agg)
        V = V + self.dropout(V_agg)

        s, V = self.update_gvp(s, V)
        s = self.norm_s(s)

        g.ndata["s"] = s
        g.ndata["V"] = V
        return g


class GVPEncoder(nn.Module):
    """GVP编码器，处理残基级图"""

    def __init__(self, node_s_dim, node_v_dim, edge_s_dim, edge_v_dim,
                 hidden_s_dim=128, hidden_v_dim=32, num_layers=3, dropout=0.1):
        super().__init__()
        self.proj_s = nn.Linear(node_s_dim, hidden_s_dim)
        self.proj_v = nn.Linear(node_v_dim, hidden_v_dim, bias=False)
        self.proj_edge_s = nn.Linear(edge_s_dim, hidden_s_dim)
        self.proj_edge_v = nn.Linear(edge_v_dim, hidden_v_dim, bias=False)
        self.norm_s_in = nn.LayerNorm(hidden_s_dim)

        self.layers = nn.ModuleList([
            GVPConvLayer(hidden_s_dim, hidden_v_dim, dropout)
            for _ in range(num_layers)
        ])
        self.readout = nn.Sequential(
            nn.Linear(hidden_s_dim, hidden_s_dim),
            nn.SiLU(),
            nn.Linear(hidden_s_dim, hidden_s_dim)
        )

    def forward(self, g):
        s = g.ndata['feat']
        V = g.ndata['vec']
        e_s = g.edata['edge_attr']
        e_V = g.edata['edge_vec']

        if V.dim() == 2:
            V = V.unsqueeze(1)
        if e_V.dim() == 2:
            e_V = e_V.unsqueeze(1)

        s = self.proj_s(s)
        s = self.norm_s_in(s)
        V = self.proj_v(V.transpose(-1, -2)).transpose(-1, -2)

        e_s = self.proj_edge_s(e_s)
        e_V = self.proj_edge_v(e_V.transpose(-1, -2)).transpose(-1, -2)

        g.ndata['s'] = s
        g.ndata['V'] = V
        g.edata['s'] = e_s
        g.edata['V'] = e_V

        for layer in self.layers:
            g = layer(g)

        return g.ndata['s']

# class GVP(nn.Module):
#     """几何向量感知器模块（Geometric Vector Perceptron）"""

#     def __init__(self, dim_s_in, dim_v_in, dim_s_out, dim_v_out, dropout=0.1):
#         super().__init__()
#         self.dim_v_in = dim_v_in
#         self.dim_v_out = dim_v_out

#         self.W_s = nn.Sequential(
#             nn.Linear(dim_s_in + dim_v_in, dim_s_out),
#             nn.SiLU()
#         )

#         self.W_v = nn.Linear(dim_v_in, dim_v_out, bias=False)

#         self.gate = nn.Sequential(
#             nn.Linear(dim_s_in + dim_v_in, dim_v_out),
#             nn.Sigmoid()
#         )

#         self.s_from_v = nn.Linear(dim_v_in, dim_v_in)
#         self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

#     def forward(self, s, V):
#         # V: (N, d_v, 3)
#         v_norm = torch.sqrt(torch.sum(V ** 2, dim=-1) + 1e-8)  # (N, d_v)

#         s_v = self.s_from_v(v_norm)
#         s_cat = torch.cat([s, s_v], dim=-1)

#         s_out = self.W_s(s_cat)

#         V_proj = self.W_v(V.transpose(-1, -2)).transpose(-1, -2)
#         gate = self.gate(s_cat).unsqueeze(-1)
#         V_out = V_proj * gate

#         return self.dropout(s_out), V_out


# class GVPSequential(nn.Module):
#     """多个 GVP 顺序连接，用于 message function 或 feed-forward function"""

#     def __init__(self, *layers):
#         super().__init__()
#         self.layers = nn.ModuleList(layers)

#     def forward(self, s, V):
#         for layer in self.layers:
#             s, V = layer(s, V)
#         return s, V


# class TupleLayerNorm(nn.Module):
#     """
#     GVP-style tuple LayerNorm.
#     标量通道做普通 LayerNorm；
#     向量通道按照向量范数进行尺度归一化，而不是对 x/y/z 分量做普通 LayerNorm。
#     """

#     def __init__(self, scalar_dim, eps=1e-8):
#         super().__init__()
#         self.scalar_norm = nn.LayerNorm(scalar_dim)
#         self.eps = eps

#     def forward(self, s, V):
#         s = self.scalar_norm(s)

#         # V: (N, d_v, 3)
#         # 每个节点计算所有向量通道范数的均方根
#         V_norm_sq = torch.sum(V ** 2, dim=-1, keepdim=True)  # (N, d_v, 1)
#         scale = torch.sqrt(torch.mean(V_norm_sq, dim=-2, keepdim=True) + self.eps)  # (N, 1, 1)

#         V = V / scale
#         return s, V


# class TupleDropout(nn.Module):
#     """
#     GVP-style dropout.
#     标量通道普通 dropout；
#     向量通道按整个 vector channel dropout，避免单独丢弃 x/y/z 分量破坏方向。
#     """

#     def __init__(self, p=0.1):
#         super().__init__()
#         self.p = p
#         self.scalar_dropout = nn.Dropout(p)

#     def forward(self, s, V):
#         s = self.scalar_dropout(s)

#         if (not self.training) or self.p == 0:
#             return s, V

#         # 对整个向量通道做 dropout，mask 形状为 (N, d_v, 1)
#         keep_prob = 1.0 - self.p
#         mask = V.new_empty(V.shape[:-1] + (1,)).bernoulli_(keep_prob)
#         mask = mask / keep_prob
#         V = V * mask

#         return s, V


# class GVPToScalar(nn.Module):
#     """
#     将最终的 (s, V) tuple 表示映射为纯标量表示。
#     这一步对应 GVP-GNN 中 node-wise GVP reducing node embeddings to scalars。
#     """

#     def __init__(self, dim_s_in, dim_v_in, dim_s_out, dropout=0.1):
#         super().__init__()

#         self.s_from_v = nn.Linear(dim_v_in, dim_v_in)
#         self.W_s = nn.Sequential(
#             nn.Linear(dim_s_in + dim_v_in, dim_s_out),
#             nn.SiLU(),
#             nn.Dropout(dropout) if dropout > 0 else nn.Identity()
#         )

#     def forward(self, s, V):
#         v_norm = torch.sqrt(torch.sum(V ** 2, dim=-1) + 1e-8)
#         s_v = self.s_from_v(v_norm)
#         s_cat = torch.cat([s, s_v], dim=-1)
#         s_out = self.W_s(s_cat)
#         return s_out


# class GVPConvLayer(nn.Module):
#     """GVP 图卷积层：message passing + point-wise feed-forward"""

#     def __init__(self, hidden_s_dim, hidden_v_dim, dropout=0.1):
#         super().__init__()

#         # 按 GVP-GNN 原文：消息由源节点 h_j 和边 h_e(j->i) 生成
#         # 因此标量输入维度是 2 * hidden_s_dim，向量输入维度是 2 * hidden_v_dim
#         self.message_gvp = GVPSequential(
#             GVP(
#                 dim_s_in=2 * hidden_s_dim,
#                 dim_v_in=2 * hidden_v_dim,
#                 dim_s_out=hidden_s_dim,
#                 dim_v_out=hidden_v_dim,
#                 dropout=dropout
#             ),
#             GVP(
#                 dim_s_in=hidden_s_dim,
#                 dim_v_in=hidden_v_dim,
#                 dim_s_out=hidden_s_dim,
#                 dim_v_out=hidden_v_dim,
#                 dropout=dropout
#             ),
#             GVP(
#                 dim_s_in=hidden_s_dim,
#                 dim_v_in=hidden_v_dim,
#                 dim_s_out=hidden_s_dim,
#                 dim_v_out=hidden_v_dim,
#                 dropout=dropout
#             )
#         )

#         # 对应原文中的 feed-forward point-wise layer，两层 GVP
#         self.ffn = GVPSequential(
#             GVP(
#                 dim_s_in=hidden_s_dim,
#                 dim_v_in=hidden_v_dim,
#                 dim_s_out=hidden_s_dim,
#                 dim_v_out=hidden_v_dim,
#                 dropout=dropout
#             ),
#             GVP(
#                 dim_s_in=hidden_s_dim,
#                 dim_v_in=hidden_v_dim,
#                 dim_s_out=hidden_s_dim,
#                 dim_v_out=hidden_v_dim,
#                 dropout=dropout
#             )
#         )

#         self.norm_msg = TupleLayerNorm(hidden_s_dim)
#         self.norm_ffn = TupleLayerNorm(hidden_s_dim)
#         self.dropout = TupleDropout(dropout)

#     def message_func(self, edges):
#         # 严格按 GVP-GNN 原文：concat(h_v(j), h_e(j -> i))
#         s_input = torch.cat(
#             [edges.src["s"], edges.data["s"]],
#             dim=-1
#         )

#         V_input = torch.cat(
#             [edges.src["V"], edges.data["V"]],
#             dim=1
#         )

#         s_msg, V_msg = self.message_gvp(s_input, V_input)

#         return {
#             "s_msg": s_msg,
#             "V_msg": V_msg
#         }

#     def reduce_func(self, nodes):
#         s_msg = nodes.mailbox["s_msg"]   # (N, deg, hidden_s)
#         V_msg = nodes.mailbox["V_msg"]   # (N, deg, hidden_v, 3)

#         # 保留你现在的入度归一化方式：1 / sqrt(deg)
#         deg = s_msg.shape[1]
#         norm = s_msg.new_tensor(deg).sqrt()

#         s_agg = s_msg.sum(dim=1) / norm
#         V_agg = V_msg.sum(dim=1) / norm

#         return {
#             "s_agg": s_agg,
#             "V_agg": V_agg
#         }

#     def forward(self, g):
#         # -------- 1. message passing --------
#         g.update_all(self.message_func, self.reduce_func)

#         s_agg = g.ndata.pop("s_agg")
#         V_agg = g.ndata.pop("V_agg")

#         s = g.ndata["s"]
#         V = g.ndata["V"]

#         # 残差连接 + Dropout + TupleLayerNorm
#         s_drop, V_drop = self.dropout(s_agg, V_agg)

#         s = s + s_drop
#         V = V + V_drop

#         s, V = self.norm_msg(s, V)

#         # -------- 2. point-wise feed-forward --------
#         s_ffn, V_ffn = self.ffn(s, V)
#         s_ffn, V_ffn = self.dropout(s_ffn, V_ffn)

#         s = s + s_ffn
#         V = V + V_ffn

#         s, V = self.norm_ffn(s, V)

#         g.ndata["s"] = s
#         g.ndata["V"] = V

#         return g


# class GVPEncoder(nn.Module):
#     """GVP 编码器，处理残基级图"""

#     def __init__(
#         self,
#         node_s_dim,
#         node_v_dim,
#         edge_s_dim,
#         edge_v_dim,
#         hidden_s_dim=128,
#         hidden_v_dim=32,
#         num_layers=3,
#         dropout=0.1
#     ):
#         super().__init__()

#         self.proj_s = nn.Linear(node_s_dim, hidden_s_dim)
#         self.proj_v = nn.Linear(node_v_dim, hidden_v_dim, bias=False)

#         self.proj_edge_s = nn.Linear(edge_s_dim, hidden_s_dim)
#         self.proj_edge_v = nn.Linear(edge_v_dim, hidden_v_dim, bias=False)

#         # 输入处也使用 tuple LayerNorm
#         self.norm_in = TupleLayerNorm(hidden_s_dim)

#         self.layers = nn.ModuleList([
#             GVPConvLayer(hidden_s_dim, hidden_v_dim, dropout)
#             for _ in range(num_layers)
#         ])

#         # 输出处使用 node-wise GVP 将 (s, V) 转为纯标量残基表示
#         self.out_norm = TupleLayerNorm(hidden_s_dim)
#         self.out_gvp = GVPToScalar(
#             dim_s_in=hidden_s_dim,
#             dim_v_in=hidden_v_dim,
#             dim_s_out=hidden_s_dim,
#             dropout=dropout
#         )

#     def forward(self, g):
#         s = g.ndata["feat"]
#         V = g.ndata["vec"]

#         e_s = g.edata["edge_attr"]
#         e_V = g.edata["edge_vec"]

#         if V.dim() == 2:
#             V = V.unsqueeze(1)

#         if e_V.dim() == 2:
#             e_V = e_V.unsqueeze(1)

#         # 节点特征投影
#         s = self.proj_s(s)
#         V = self.proj_v(V.transpose(-1, -2)).transpose(-1, -2)

#         # 边特征投影
#         e_s = self.proj_edge_s(e_s)
#         e_V = self.proj_edge_v(e_V.transpose(-1, -2)).transpose(-1, -2)

#         # 输入 tuple normalization
#         s, V = self.norm_in(s, V)

#         g.ndata["s"] = s
#         g.ndata["V"] = V
#         g.edata["s"] = e_s
#         g.edata["V"] = e_V

#         for layer in self.layers:
#             g = layer(g)

#         # 输出前再次 tuple normalization
#         s, V = self.out_norm(g.ndata["s"], g.ndata["V"])

#         # 将最终 (s, V) 映射为纯标量残基表示
#         h_res = self.out_gvp(s, V)

#         return h_res
# =========================================================
# 3. Projection Head
# =========================================================
class ProjectionHead(nn.Module):
    """对比学习投影头：将编码特征映射到低维空间并L2归一化"""

    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        x = self.net(x)
        return F.normalize(x, dim=-1)


# =========================================================
# 4. Atom Reconstruction Decoder (MAE style)
# =========================================================
class AtomReconstructionDecoder(nn.Module):
    """从残基特征与原子编码重建被遮蔽原子的类型与坐标"""

    def __init__(self, hidden_dim, num_atom_types):
        super().__init__()
        self.res_to_atom = nn.Linear(hidden_dim, hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.type_predictor = nn.Linear(hidden_dim, num_atom_types)
        self.coord_predictor = nn.Linear(hidden_dim, 3)

    def forward(self, h_res, h_atom, atom_to_resid, x_ca, R_res):
        # h_res: (N_res, hidden_dim)
        # h_atom: (N_atom, hidden_dim) — 所有原子的编码特征
        # atom_to_resid: (N_atom,)
        # x_ca: (N_res, 3)
        # R_res: (N_res, 3, 3)

        # 将残基特征广播到对应原子
        h_cond = self.res_to_atom(h_res)[atom_to_resid]   # (N_atom, hidden)

        # 拼接条件特征与编码原子特征
        h = self.mlp(torch.cat([h_cond, h_atom], dim=-1))

        # 预测原子类型与局部偏移
        type_logits = self.type_predictor(h)
        delta_local = self.coord_predictor(h)

        # 将局部坐标转换到全局坐标系
        R_atom = R_res[atom_to_resid]
        delta_global = torch.einsum('nij,nj->ni', R_atom, delta_local)
        x_pred = x_ca[atom_to_resid] + delta_global

        return type_logits, x_pred


# =========================================================
# 5. Full Pretraining Model
# =========================================================
class ARBindNet_Pretrain(nn.Module):
    """ARBindNet预训练模型：结合对比学习与掩码重建"""

    def __init__(self, config):
        super().__init__()
        self.config = config

        # 原子编码器 (EGNN)
        self.atom_encoder = EGNNEncoder(
            node_dim=config.atom_node_dim,
            edge_dim=config.atom_edge_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.egnn_layers,
            num_heads=config.egnn_heads,
            use_attention=config.use_attention,
            coord_noise_std=getattr(config, 'coord_noise_std', 0.3)
        )

        # 残基编码器 (GVP)
        self.res_encoder = GVPEncoder(
            node_s_dim=config.res_node_s_dim,
            node_v_dim=config.res_node_v_dim,
            edge_s_dim=config.res_edge_s_dim,
            edge_v_dim=config.res_edge_v_dim,
            hidden_s_dim=config.gvp_hidden_s,
            hidden_v_dim=config.gvp_hidden_v,
            num_layers=config.gvp_layers,
            dropout=config.gvp_dropout
        )

        # 原子特征到残基的注意力聚合
        self.atom_to_res_attn = nn.MultiheadAttention(
            embed_dim=config.hidden_dim,
            num_heads=4,
            batch_first=True
        )
        self.res_query = nn.Parameter(torch.randn(1, 1, config.hidden_dim))

        # 投影头（对比学习）
        self.proj_atom = ProjectionHead(config.hidden_dim, config.proj_hidden, config.proj_out)
        self.proj_res = ProjectionHead(config.gvp_hidden_s, config.proj_hidden, config.proj_out)

        # 重建解码器
        self.decoder = AtomReconstructionDecoder(
            hidden_dim=config.hidden_dim,
            num_atom_types=config.num_atom_types
        )

        self.mask_ratio = config.mask_ratio
        self.temperature = config.temperature

    def forward(self, atom_g, res_g, atom_to_resid):
        device = atom_g.device
        N_atom = atom_g.num_nodes()

        # -------- 1. 随机掩码 --------
        mask = random_mask_atom(N_atom, self.mask_ratio, device)

        # -------- 2. 原子编码（带掩码标记与噪声坐标）--------
        h_atom, x_atom = self.atom_encoder(atom_g, mask)

        # -------- 3. 残基编码 --------
        h_res_node = self.res_encoder(res_g)
        x_ca = res_g.ndata['x']
        R_res = res_g.ndata['frame']
        batch_atom = torch.repeat_interleave(
            torch.arange(len(atom_g.batch_num_nodes()), device=device),
            atom_g.batch_num_nodes()
        )
        # -------- 4. 可见原子特征聚合到残基 --------
        visible_mask = ~mask
        h_atom_visible = h_atom[visible_mask]
        batch_atom_visible = batch_atom[visible_mask]
        # atom_to_resid_visible = atom_to_resid[visible_mask]
        # num_res = res_g.num_nodes()
        # h_res_atom = self._attn_pool_to_residue(h_atom_visible, atom_to_resid_visible, num_res)

        # -------- 5. 投影到对比空间 --------
        z_atom = self.proj_atom(h_atom_visible)
        z_res = self.proj_res(h_res_node)

        # -------- 6. 图级平均池化（用于对比损失）--------
        batch_res = torch.repeat_interleave(
            torch.arange(len(res_g.batch_num_nodes()), device=device),
            res_g.batch_num_nodes()
        )
        z_atom_graph = self._avg_pool_per_graph(z_atom, batch_atom_visible)
        z_res_graph = self._avg_pool_per_graph(z_res, batch_res)

        # -------- 7. 重建被遮蔽原子 --------
        type_logits, x_pred = self.decoder(
            h_res_node,          # 残基特征
            h_atom,              # 所有原子编码特征
            atom_to_resid,
            x_ca,
            R_res
        )

        return z_atom_graph, z_res_graph, type_logits, x_pred, mask

    def _avg_pool_per_graph(self, x, batch_idx):
        num_graphs = batch_idx.max().item() + 1
        pooled = torch.zeros(num_graphs, x.size(1), device=x.device)
        pooled.index_add_(0, batch_idx, x)
        count = torch.bincount(batch_idx, minlength=num_graphs).float().clamp(min=1).unsqueeze(-1)
        return pooled / count

    def _attn_pool_to_residue(self, h_atom, atom_to_resid, num_res):
        """
        使用点积注意力将原子特征汇聚到残基级。
        h_atom: (N_visible, hidden_dim)
        atom_to_resid: (N_visible,)
        """
        query = self.res_query.squeeze(0).expand(num_res, -1)   # (num_res, hidden_dim)
        query_per_atom = query[atom_to_resid]                   # (N_visible, hidden_dim)
        scores = (h_atom * query_per_atom).sum(dim=-1)          # (N_visible,)
        attn_weights = scatter_softmax(scores, atom_to_resid, dim=0)
        weighted = h_atom * attn_weights.unsqueeze(-1)
        res_feat = scatter_sum(weighted, atom_to_resid, dim=0, dim_size=num_res)
        return res_feat