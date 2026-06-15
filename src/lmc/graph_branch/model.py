#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import Tuple
import math

import torch
from typing import List
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_align
import dgl  # type: ignore
from dgl.nn import GraphConv, GATConv, GINConv  # type: ignore
from .backbone import build_backbone


class TemporalEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.gru = nn.GRU(input_size=in_dim, hidden_size=hidden_dim, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, h = self.gru(x)
        return h[-1]


class DGLGCNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.conv = GraphConv(in_dim, out_dim, norm="both", weight=True, bias=True, allow_zero_in_degree=True)

    @staticmethod
    def _adj_to_graph(adj: torch.Tensor):
        src, dst = torch.where(adj > 0)
        g = dgl.graph((src, dst), num_nodes=adj.size(0), device=adj.device)
        g = dgl.add_self_loop(g)
        return g

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        g = self._adj_to_graph(adj)
        return self.conv(g, x)


class DGLGATLayer(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_heads: int = 1,
        feat_drop: float = 0.0,
        attn_drop: float = 0.0,
        residual: bool = False,
    ):
        super().__init__()
        if out_dim % int(num_heads) != 0:
            raise ValueError(f"out_dim ({out_dim}) must be divisible by num_heads ({num_heads})")
        out_per_head = out_dim // int(num_heads)
        self.conv = GATConv(
            in_dim,
            out_per_head,
            num_heads=int(num_heads),
            feat_drop=float(feat_drop),
            attn_drop=float(attn_drop),
            negative_slope=0.2,
            residual=bool(residual),
            activation=None,
            allow_zero_in_degree=True,
        )

    @staticmethod
    def _adj_to_graph(adj: torch.Tensor):
        src, dst = torch.where(adj > 0)
        g = dgl.graph((src, dst), num_nodes=adj.size(0), device=adj.device)
        g = dgl.add_self_loop(g)
        return g

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        g = self._adj_to_graph(adj)
        y = self.conv(g, x)  # (N, heads, out_per_head)
        return y.flatten(1)


class DGLGINLayer(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        aggregator_type: str = "sum",
        init_eps: float = 0.0,
        learn_eps: bool = True,
    ):
        super().__init__()
        apply_func = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(out_dim, out_dim),
        )
        self.conv = GINConv(
            apply_func=apply_func,
            aggregator_type=str(aggregator_type),
            init_eps=float(init_eps),
            learn_eps=bool(learn_eps),
        )

    @staticmethod
    def _adj_to_graph(adj: torch.Tensor):
        src, dst = torch.where(adj > 0)
        g = dgl.graph((src, dst), num_nodes=adj.size(0), device=adj.device)
        g = dgl.add_self_loop(g)
        return g

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        g = self._adj_to_graph(adj)
        return self.conv(g, x)


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 128, layers: int = 2):
        super().__init__()
        dims = [in_dim] + [hidden] * max(0, layers - 1) + [out_dim]
        modules = []
        for i in range(len(dims) - 1):
            modules.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                modules.append(nn.ReLU(inplace=True))
        self.net = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EdgeGraphNet(nn.Module):
    def __init__(
        self,
        dyn_dim: int,
        morph_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dyn_mlp_layers: int = 2,
        morph_mlp_layers: int = 2,
        gnn_type: str = "gcn",
        gat_num_heads: int = 1,
        gat_feat_drop: float = 0.0,
        gat_attn_drop: float = 0.0,
        gat_residual: bool = False,
        gin_aggregator_type: str = "sum",
        gin_init_eps: float = 0.0,
        gin_learn_eps: bool = True,
        use_bbox_hw: bool = False,
        layer_concat_enable: bool = True,
    ):
        super().__init__()
        self.gnn_type = str(gnn_type).lower()
        self.use_bbox_hw = bool(use_bbox_hw)
        self.layer_concat_enable = bool(layer_concat_enable)
        self.num_layers = int(num_layers)
        self.dyn_mlp = MLP(dyn_dim, hidden_dim, hidden=hidden_dim, layers=dyn_mlp_layers)
        self.morph_dim = int(morph_dim)
        if self.morph_dim > 0:
            self.morph_mlp = MLP(morph_dim, hidden_dim, hidden=hidden_dim, layers=morph_mlp_layers)
            self.fuse_mlp = MLP(hidden_dim * 2, hidden_dim, hidden=hidden_dim, layers=2)
        else:
            self.morph_mlp = None
            self.fuse_mlp = MLP(hidden_dim, hidden_dim, hidden=hidden_dim, layers=2)
        self.temporal = TemporalEncoder(hidden_dim, hidden_dim)

        gnn_layers = []
        self.geo_fuse_layers = nn.ModuleList()
        for i in range(num_layers):
            gnn_in = hidden_dim
            if self.gnn_type == "gat":
                gnn_layers.append(
                    DGLGATLayer(
                        gnn_in,
                        hidden_dim,
                        num_heads=gat_num_heads,
                        feat_drop=gat_feat_drop,
                        attn_drop=gat_attn_drop,
                        residual=gat_residual,
                    )
                )
            elif self.gnn_type == "gin":
                gnn_layers.append(
                    DGLGINLayer(
                        gnn_in,
                        hidden_dim,
                        aggregator_type=gin_aggregator_type,
                        init_eps=gin_init_eps,
                        learn_eps=gin_learn_eps,
                    )
                )
            else:
                gnn_layers.append(DGLGCNLayer(gnn_in, hidden_dim))
            gnn_layers.append(nn.ReLU(inplace=True))
            self.geo_fuse_layers.append(MLP(hidden_dim * 2, hidden_dim, hidden=hidden_dim, layers=2))
        self.gnn = nn.Sequential(*gnn_layers)
        self.classifier_last = nn.Linear(hidden_dim, 1)
        self.concat_classifier = MLP(hidden_dim * self.num_layers, 1, hidden=hidden_dim, layers=2)

    def forward(
        self,
        x_dyn: torch.Tensor,
        x_morph: torch.Tensor,
        adj: torch.Tensor,
        geo_feat: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        n, t, _ = x_dyn.shape
        dyn = self.dyn_mlp(x_dyn.reshape(n * t, -1)).reshape(n, t, -1)
        if self.morph_dim > 0:
            morph = self.morph_mlp(x_morph)
            morph = morph[:, None, :].expand(-1, t, -1)
            fused = self.fuse_mlp(torch.cat([dyn, morph], dim=-1).reshape(n * t, -1)).reshape(n, t, -1)
        else:
            fused = self.fuse_mlp(dyn.reshape(n * t, -1)).reshape(n, t, -1)

        h = self.temporal(fused)
        if geo_feat is None:
            geo_feat = h.new_zeros(h.shape)
        conv_idx = 0
        layer_node_feats: List[torch.Tensor] = []
        for layer in self.gnn:
            if isinstance(layer, (DGLGCNLayer, DGLGATLayer, DGLGINLayer)):
                if self.use_bbox_hw:
                    h = self.geo_fuse_layers[conv_idx](torch.cat([h, geo_feat], dim=-1))
                h = layer(h, adj)
                conv_idx += 1
            else:
                h = layer(h)
                if isinstance(layer, nn.ReLU):
                    layer_node_feats.append(h)
        if len(layer_node_feats) == 0:
            layer_node_feats = [h]
        if self.layer_concat_enable:
            h_cat = torch.cat(layer_node_feats, dim=-1)
            logits = self.concat_classifier(h_cat).squeeze(-1)
        else:
            logits = self.classifier_last(h).squeeze(-1)
        return logits, h, layer_node_feats


class EdgeGraphFeatureNet(nn.Module):
    def __init__(
        self,
        morph_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dyn_mlp_layers: int = 2,
        morph_mlp_layers: int = 2,
        gnn_type: str = "gcn",
        ablate_roi: bool = False,
        ablate_center: bool = False,
        ablate_pos: bool = False,
        use_pos_embed: bool = False,
        use_vessel_mask: bool = True,
        roi_mode: str = "bbox",
        roi_pool: str = "mean",
        use_bbox_hw: bool = False,
        gat_num_heads: int = 1,
        gat_feat_drop: float = 0.0,
        gat_attn_drop: float = 0.0,
        gat_residual: bool = False,
        gin_aggregator_type: str = "sum",
        gin_init_eps: float = 0.0,
        gin_learn_eps: bool = True,
        layer_concat_enable: bool = True,
        backbone_type: str = "unet",
        dino_model_name: str = "dinov3_vitb16",
        dino_repo_dir: str = "/home/juncao/data/app/dinov3",
        dino_backbone_weights: str = "",
        dino_freeze: bool = True,
        dino_use_autocast_bf16: bool = True,
        dino_gray_to_rgb: bool = True,
        dino_layer_select: int = -1,
        pos_freq: int = 10,
    ):
        super().__init__()
        self.backbone = build_backbone(
            backbone_type=backbone_type,
            hidden_dim=int(hidden_dim),
            dino_model_name=dino_model_name,
            dino_repo_dir=dino_repo_dir,
            dino_backbone_weights=dino_backbone_weights,
            dino_freeze=bool(dino_freeze),
            dino_use_autocast_bf16=bool(dino_use_autocast_bf16),
            dino_gray_to_rgb=bool(dino_gray_to_rgb),
            dino_layer_select=int(dino_layer_select),
        )
        self.backbone_feat_dim = int(getattr(self.backbone, "out_dim", 64))
        self.ablate_roi = ablate_roi
        self.ablate_center = ablate_center
        self.ablate_pos = ablate_pos
        self.use_pos_embed = use_pos_embed
        self.use_vessel_mask = bool(use_vessel_mask)
        self.roi_mode = roi_mode
        self.roi_pool = str(roi_pool).lower()
        self.use_bbox_hw = bool(use_bbox_hw)
        self.roi_token_size = 7
        # Token-style ROI pooling head: Cx7x7 -> C, keeps local texture better than 1x1 mean pooling.
        self.roi_token_head = nn.Sequential(
            nn.Conv2d(self.backbone_feat_dim, self.backbone_feat_dim, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(self.backbone_feat_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.backbone_feat_dim, self.backbone_feat_dim, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(self.backbone_feat_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        dyn_dim = int(self.backbone_feat_dim) + 1  # backbone feature + point feature from logits
        # positional encoding (center_proj) -> MLP -> add to node features
        # Classic transformer-style sin/cos encoding on x and y (d/2 each).
        self.pos_freq = int(pos_freq)
        self.pos_dim = 2 * self.pos_freq  # total dim; x uses d/2, y uses d/2
        if self.pos_dim % 4 != 0:
            raise ValueError(f"pos_dim must be divisible by 4 for x/y sincos, got {self.pos_dim}")
        self.pos_mlp = MLP(self.pos_dim, dyn_dim, hidden=hidden_dim, layers=2) if self.use_pos_embed else None
        self.hw_mlp = MLP(2, dyn_dim, hidden=hidden_dim, layers=2) if self.use_bbox_hw else None
        self.geo_fuse_mlp = MLP(dyn_dim * 2, hidden_dim, hidden=hidden_dim, layers=2) if self.use_bbox_hw else None
        self.roi_meanmax_proj = nn.Linear(self.backbone_feat_dim * 2, self.backbone_feat_dim)
        self.morph_dim = int(morph_dim)
        self.graph = EdgeGraphNet(
            dyn_dim,
            morph_dim,
            hidden_dim,
            num_layers,
            dyn_mlp_layers=dyn_mlp_layers,
            morph_mlp_layers=morph_mlp_layers,
            gnn_type=gnn_type,
            gat_num_heads=gat_num_heads,
            gat_feat_drop=gat_feat_drop,
            gat_attn_drop=gat_attn_drop,
            gat_residual=gat_residual,
            gin_aggregator_type=gin_aggregator_type,
            gin_init_eps=gin_init_eps,
            gin_learn_eps=gin_learn_eps,
            use_bbox_hw=self.use_bbox_hw,
            layer_concat_enable=layer_concat_enable,
        )

    def _positional_encoding(self, centers: torch.Tensor, h: int, w: int) -> torch.Tensor:
        # centers: (N,2) [y,x] in pixel coordinates
        y = centers[:, 0].to(dtype=torch.float32)
        x = centers[:, 1].to(dtype=torch.float32)
        n = centers.size(0)
        half_dim = self.pos_dim // 2  # x uses d/2, y uses d/2
        if half_dim % 2 != 0:
            raise ValueError(f"pos_dim/2 must be even, got {half_dim}")
        div_term = torch.exp(
            torch.arange(0, half_dim, 2, device=centers.device, dtype=torch.float32)
            * (-math.log(10000.0) / half_dim)
        )
        pe_x = torch.zeros((n, half_dim), device=centers.device, dtype=torch.float32)
        pe_y = torch.zeros((n, half_dim), device=centers.device, dtype=torch.float32)
        pe_x[:, 0::2] = torch.sin(x[:, None] * div_term)
        pe_x[:, 1::2] = torch.cos(x[:, None] * div_term)
        pe_y[:, 0::2] = torch.sin(y[:, None] * div_term)
        pe_y[:, 1::2] = torch.cos(y[:, None] * div_term)
        pos = torch.cat([pe_x, pe_y], dim=1)
        return pos

    def _roi_features(self, feat: torch.Tensor, bboxes: torch.Tensor) -> torch.Tensor:
        # feat: (T,C,H,W), bboxes: (N,4) [y0,y1,x0,x1]
        t, c, h, w = feat.shape
        n = bboxes.shape[0]
        bbox_rep = bboxes[None].repeat(t, 1, 1).reshape(-1, 4)
        batch_idx = torch.arange(t, device=feat.device).repeat_interleave(n).float()
        y0 = bbox_rep[:, 0]
        y1 = bbox_rep[:, 1] + 1.0
        x0 = bbox_rep[:, 2]
        x1 = bbox_rep[:, 3] + 1.0
        rois = torch.stack([batch_idx, x0, y0, x1, y1], dim=1)
        if self.roi_pool in ("max", "meanmax"):
            roi_raw = roi_align(
                feat,
                rois,
                output_size=(self.roi_token_size, self.roi_token_size),
                spatial_scale=1.0,
                sampling_ratio=2,
                aligned=True,
            )
            roi_avg = F.adaptive_avg_pool2d(roi_raw, 1).view(t, n, c).permute(1, 0, 2)
            roi_max = F.adaptive_max_pool2d(roi_raw, 1).view(t, n, c).permute(1, 0, 2)
            if self.roi_pool == "max":
                roi_feat = roi_max
            else:
                roi_feat = self.roi_meanmax_proj(torch.cat([roi_avg, roi_max], dim=-1))
        else:
            roi_feat = roi_align(feat, rois, output_size=(1, 1), spatial_scale=1.0, sampling_ratio=2, aligned=True)
            roi_feat = roi_feat.view(t, n, c).permute(1, 0, 2)  # (N,T,C)
        return roi_feat

    def _bbox_hw_features(self, bboxes: torch.Tensor, h: int, w: int) -> torch.Tensor:
        # bboxes: (N,4) [y0,y1,x0,x1]
        bh = (bboxes[:, 1] - bboxes[:, 0] + 1.0).clamp(min=1.0) / 100.0
        bw = (bboxes[:, 3] - bboxes[:, 2] + 1.0).clamp(min=1.0) / 100.0
        return torch.stack([bh, bw], dim=1)

    def _roi_token_features(self, feat: torch.Tensor, bboxes: torch.Tensor) -> torch.Tensor:
        # feat: (T,C,H,W), bboxes: (N,4) [y0,y1,x0,x1]
        t, c, h, w = feat.shape
        n = bboxes.shape[0]
        bbox_rep = bboxes[None].repeat(t, 1, 1).reshape(-1, 4)
        batch_idx = torch.arange(t, device=feat.device).repeat_interleave(n).float()
        y0 = bbox_rep[:, 0]
        y1 = bbox_rep[:, 1] + 1.0
        x0 = bbox_rep[:, 2]
        x1 = bbox_rep[:, 3] + 1.0
        rois = torch.stack([batch_idx, x0, y0, x1, y1], dim=1)
        roi_tok = roi_align(
            feat,
            rois,
            output_size=(self.roi_token_size, self.roi_token_size),
            spatial_scale=1.0,
            sampling_ratio=2,
            aligned=True,
        )  # (T*N,C,K,K)
        roi_tok = self.roi_token_head(roi_tok).view(t, n, c).permute(1, 0, 2)  # (N,T,C)
        return roi_tok

    def _pixel_features(self, feat: torch.Tensor, pixels: List[torch.Tensor]) -> torch.Tensor:
        # feat: (T,C,H,W), pixels: list of (K,2) [y,x]
        t, c, h, w = feat.shape
        feats = []
        for pix in pixels:
            if pix.numel() == 0:
                feats.append(feat.new_zeros((t, c)))
                continue
            p = pix.to(feat.device).long()
            ys = p[:, 0].clamp(0, h - 1)
            xs = p[:, 1].clamp(0, w - 1)
            f = feat[:, :, ys, xs]  # (T,C,K)
            feats.append(f.mean(dim=2))
        return torch.stack(feats, dim=0)  # (N,T,C)

    def _point_features(self, logits: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
        # logits: (T,1,H,W), centers: (N,2) [y,x]
        t, _, h, w = logits.shape
        n = centers.shape[0]
        y_norm = (centers[:, 0] / max(1.0, (h - 1))) * 2.0 - 1.0
        x_norm = (centers[:, 1] / max(1.0, (w - 1))) * 2.0 - 1.0
        grid = torch.stack([x_norm, y_norm], dim=-1).view(1, n, 1, 2).repeat(t, 1, 1, 1)
        vals = F.grid_sample(logits, grid, align_corners=True)  # (T,1,N,1)
        vals = vals.permute(2, 0, 1, 3).squeeze(-1)  # (N,T,1)
        return vals

    def forward(
        self,
        frames: torch.Tensor,
        bboxes: torch.Tensor,
        centers: torch.Tensor,
        x_morph: torch.Tensor,
        adj: torch.Tensor,
        edge_index: torch.Tensor,
        vessel_masks: torch.Tensor = None,
        pixels: List[torch.Tensor] = None,
        return_aux: bool = False,
    ):
        device = frames.device
        if self.use_vessel_mask and (vessel_masks is not None):
            vessel_masks = vessel_masks.to(device)
            frames = frames * vessel_masks
        feat, logits, _ = self.backbone.forward_features(frames)
        pos_feat = None
        if self.use_pos_embed and (not self.ablate_pos):
            # positional embedding from center_proj
            pos = self._positional_encoding(centers.to(device), logits.shape[2], logits.shape[3])
            pos_feat = self.pos_mlp(pos)
        geo_feat = None
        if self.use_bbox_hw:
            hw = self._bbox_hw_features(bboxes.to(device), logits.shape[2], logits.shape[3])
            hw_feat = self.hw_mlp(hw)
            pos_for_geo = pos_feat if pos_feat is not None else hw_feat.new_zeros(hw_feat.shape)
            geo_feat = self.geo_fuse_mlp(torch.cat([pos_for_geo, hw_feat], dim=-1))
        if self.roi_mode == "pixel":
            if pixels is None:
                raise ValueError("roi_mode='pixel' requires pixels list")
            roi_feat = self._pixel_features(feat, pixels)
        elif self.roi_mode == "bbox_token":
            roi_feat = self._roi_token_features(feat, bboxes)
        else:
            roi_feat = self._roi_features(feat, bboxes)
        point_feat = self._point_features(logits, centers)
        if self.ablate_roi:
            roi_feat = torch.zeros_like(roi_feat)
        if self.ablate_center:
            point_feat = torch.zeros_like(point_feat)
        x_dyn = torch.cat([roi_feat, point_feat], dim=-1)
        # add positional embedding to each node feature (broadcast across time)
        if pos_feat is not None:
            x_dyn = x_dyn + pos_feat[:, None, :]
        if self.morph_dim > 0:
            x_morph = x_morph.to(device)
        else:
            x_morph = x_morph.new_zeros((x_dyn.shape[0], 0))
        adj = adj.to(device)
        node_logits, node_emb_last, node_emb_layers = self.graph(x_dyn, x_morph, adj, geo_feat=geo_feat)
        if return_aux:
            return {
                "node_logits": node_logits,
                "node_emb_last": node_emb_last,
                "node_emb_layers": node_emb_layers,
            }
        return node_logits
