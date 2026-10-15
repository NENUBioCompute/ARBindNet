import torch
import torch.nn as nn
from torch_scatter import scatter_softmax, scatter_sum

from model import EGNNEncoder, GVPEncoder
from fusion import CrossLevelFusion


class PLiSAGE_LigandMulticlass(nn.Module):
    def __init__(self, config, num_classes=6, freeze_backbone=False):
        super().__init__()
        self.config = config
        self.num_classes = num_classes

        self.atom_encoder = EGNNEncoder(
            node_dim=config.atom_node_dim,
            edge_dim=config.atom_edge_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.egnn_layers,
            num_heads=config.egnn_heads,
            use_attention=config.use_attention,
        )
        self.res_encoder = GVPEncoder(
            node_s_dim=config.res_node_s_dim,
            node_v_dim=config.res_node_v_dim,
            edge_s_dim=config.res_edge_s_dim,
            edge_v_dim=config.res_edge_v_dim,
            hidden_s_dim=config.gvp_hidden_s,
            hidden_v_dim=config.gvp_hidden_v,
            num_layers=config.gvp_layers,
            dropout=config.gvp_dropout,
        )
        self.res_proj = nn.Linear(config.gvp_hidden_s, config.hidden_dim)
        self.res_query = nn.Parameter(torch.randn(1, 1, config.hidden_dim))
        self.fusion = CrossLevelFusion(hidden_dim=config.hidden_dim, num_heads=4, dropout=0.1)
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.BatchNorm1d(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.classifier_dropout),
            nn.Linear(config.hidden_dim, num_classes),
        )
        self._init_classifier_weights()

        if freeze_backbone:
            for name, param in self.named_parameters():
                if not name.startswith("classifier."):
                    param.requires_grad = False

    def _init_classifier_weights(self):
        for module in self.classifier:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def load_pretrained_backbone(self, checkpoint_path, map_location="cpu"):
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint

        own_state = self.state_dict()
        loadable = {}
        skipped = []
        for key, value in state_dict.items():
            if key.startswith("classifier."):
                skipped.append(key)
                continue
            if key in own_state and own_state[key].shape == value.shape:
                loadable[key] = value
            else:
                skipped.append(key)

        missing, unexpected = self.load_state_dict(loadable, strict=False)
        print(f"[INFO] loaded pretrained backbone tensors: {len(loadable)}")
        print(f"[INFO] skipped tensors: {len(skipped)}")
        if missing:
            print(f"[INFO] missing keys after partial load: {len(missing)}")
        if unexpected:
            print(f"[INFO] unexpected keys after partial load: {len(unexpected)}")

    def encode(self, atom_g, res_g, atom_to_resid):
        h_atom, _ = self.atom_encoder(atom_g, mask=None)
        h_res_raw = self.res_encoder(res_g)
        h_res = self.res_proj(h_res_raw)

        num_res = res_g.num_nodes()
        h_atom_pooled = self._attn_pool_to_residue(h_atom, atom_to_resid, num_res)
        batch_res = torch.repeat_interleave(
            torch.arange(len(res_g.batch_num_nodes()), device=atom_g.device),
            res_g.batch_num_nodes(),
        )
        _, h_graph = self.fusion(h_atom_pooled, h_res, batch_res)
        return h_graph

    def forward(self, atom_g, res_g, atom_to_resid):
        return self.classifier(self.encode(atom_g, res_g, atom_to_resid))

    def _attn_pool_to_residue(self, h_atom, atom_to_resid, num_res):
        query = self.res_query.squeeze(0).expand(num_res, -1)
        query_per_atom = query[atom_to_resid]
        scores = (h_atom * query_per_atom).sum(dim=-1)
        attn_weights = scatter_softmax(scores, atom_to_resid, dim=0)
        weighted = h_atom * attn_weights.unsqueeze(-1)
        return scatter_sum(weighted, atom_to_resid, dim=0, dim_size=num_res)
