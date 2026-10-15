import torch
import torch.nn as nn
from torch_scatter import scatter_softmax, scatter_sum, scatter_max

# 假设你的原始编码器和融合模块导入不变
from model import EGNNEncoder, GVPEncoder
from fusion import CrossLevelFusion


class ARBindNet_Classifier(nn.Module):
    def __init__(self, config, pretrained_path=None, freeze_encoders=False):
        super().__init__()
        self.config = config

        # ------------------- 编码器与投影 (保持不变) -------------------
        self.atom_encoder = EGNNEncoder(
            node_dim=config.atom_node_dim,
            edge_dim=config.atom_edge_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.egnn_layers,
            num_heads=config.egnn_heads,
            use_attention=config.use_attention
        )
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
        self.res_proj = nn.Linear(config.gvp_hidden_s, config.hidden_dim)
        self.res_query = nn.Parameter(torch.randn(1, 1, config.hidden_dim))
        # ------------------- 加载预训练权重 -------------------
        if pretrained_path is not None:
            self._load_pretrained(pretrained_path)
            print(f"✅ Loaded pretrained weights from {pretrained_path}")
        # ------------------- 冻结编码器 -------------------
        if freeze_encoders:
            for param in self.atom_encoder.parameters():
                param.requires_grad = False
            for param in self.res_encoder.parameters():
                param.requires_grad = False
            print("🔒 Encoders frozen")
        # ------------------- 融合模块 -------------------
        self.fusion = CrossLevelFusion(
            hidden_dim=config.hidden_dim,
            num_heads=4,
            dropout=0.1
        )

        # ------------------- 改进后的分类头 -------------------
        # # 因为使用了 Mean + Max 拼接池化，输入维度变为 hidden_dim * 2
        # self.classifier = nn.Sequential(
        #     nn.Linear(config.hidden_dim * 2, config.hidden_dim * 2),
        #     nn.BatchNorm1d(config.hidden_dim * 2),
        #     nn.GELU(),  # GELU 在此类任务中表现通常优于 ReLU
        #     nn.Dropout(config.classifier_dropout),
        #
        #     nn.Linear(config.hidden_dim * 2, config.hidden_dim),
        #     nn.GELU(),
        #     nn.Dropout(config.classifier_dropout),
        #
        #     nn.Linear(config.hidden_dim, 2)
        # )
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.BatchNorm1d(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.classifier_dropout),
            nn.Linear(config.hidden_dim, 1)
        )

        # 初始化新增参数
        self._init_classifier_weights()

        if pretrained_path is not None:
            self._load_pretrained(pretrained_path)

        if freeze_encoders:
            for param in self.atom_encoder.parameters():
                param.requires_grad = False
            for param in self.res_encoder.parameters():
                param.requires_grad = False

    def _init_classifier_weights(self):
        for m in self.classifier:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def _load_pretrained(self, path):
        checkpoint = torch.load(path, map_location='cpu')
        state_dict = checkpoint['model_state_dict']
        enc_state = {}
        for k, v in state_dict.items():
            if k.startswith('atom_encoder.') or k.startswith('res_encoder.'):
                enc_state[k] = v
        missing, unexpected = self.load_state_dict(enc_state, strict=False)
        if missing:
            print(f"Missing keys (will be randomly initialized): {missing[:5]}...")
        if unexpected:
            print(f"Unexpected keys (ignored): {unexpected[:5]}...")

    def forward(self, atom_g, res_g, atom_to_resid):
        # 1. 编码
        h_atom, _ = self.atom_encoder(atom_g, mask=None)
        h_res_raw = self.res_encoder(res_g)
        h_res = self.res_proj(h_res_raw)

        # 2. 原子到残基池化
        num_res = res_g.num_nodes()
        h_atom_pooled = self._attn_pool_to_residue(h_atom, atom_to_resid, num_res)

        # 3. 获取批索引
        batch_res = torch.repeat_interleave(
            torch.arange(len(res_g.batch_num_nodes()), device=atom_g.device),
            res_g.batch_num_nodes()
        )

        # 4. 融合 (注意：这里取融合后的残基特征进行池化)
        h_fused, h_graph = self.fusion(h_atom_pooled, h_res, batch_res)

        # # 5. 混合池化 (Mean + Max)
        # # h_fused 形状为 (N_total_res, hidden_dim)
        # h_avg = scatter_sum(h_fused, batch_res, dim=0) / res_g.batch_num_nodes().view(-1, 1)
        # h_max, _ = scatter_max(h_fused, batch_res, dim=0)
        #
        # # 拼接特征: (Batch, hidden_dim * 2)
        # h_combined = torch.cat([h_avg, h_max], dim=-1)

        # 6. 分类
        logits = self.classifier(h_graph)
        return logits

    def _attn_pool_to_residue(self, h_atom, atom_to_resid, num_res):
        # 保持你原有的注意力池化逻辑
        query = self.res_query.squeeze(0).expand(num_res, -1)
        query_per_atom = query[atom_to_resid]
        scores = (h_atom * query_per_atom).sum(dim=-1)
        attn_weights = scatter_softmax(scores, atom_to_resid, dim=0)
        weighted = h_atom * attn_weights.unsqueeze(-1)
        res_feat = scatter_sum(weighted, atom_to_resid, dim=0, dim_size=num_res)
        return res_feat
