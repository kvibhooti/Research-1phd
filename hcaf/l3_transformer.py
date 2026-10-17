import torch
import torch.nn as nn
import torchvision.models as models

class HCAFSemanticTransformer(nn.Module):
    def __init__(self, d_model=256, num_heads=8, num_layers=4, num_predicates=7, num_zones=6, num_scenes=5, num_classes=10):
        super().__init__()
        self.d_model = d_model

        # Real CNN Backbone for patch feature extraction
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        # remove fc and avgpool
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.feat_proj = nn.Conv2d(512, d_model, kernel_size=1)

        # Class embedding & Box embedding
        self.class_emb = nn.Embedding(num_classes, d_model)
        self.box_emb = nn.Linear(4, d_model)

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.relation_head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, num_predicates)
        )
        self.region_head = nn.Linear(d_model, num_zones)
        self.scene_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, num_scenes)
        )

    def forward(self, images, boxes_list, classes_list, edges_list):
        # images: (B, 3, H, W)
        bs = images.size(0)

        # Extract features from CNN
        # (B, 512, H', W')
        feature_maps = self.backbone(images)
        # (B, d_model, H', W')
        feature_maps = self.feat_proj(feature_maps)

        scene_logits_list = []
        region_logits_list = []
        relation_logits_list = []

        # Process each item in the batch sequentially due to variable number of tokens (objects)
        for i in range(bs):
            boxes = boxes_list[i] # (N, 4)
            classes = classes_list[i] # (N,)
            edges = edges_list[i] # (2, E)

            N = boxes.size(0)
            if N == 0:
                # Handle empty image
                scene_logits_list.append(self.scene_head(torch.zeros(1, self.d_model, device=images.device)))
                region_logits_list.append(self.region_head(torch.zeros(9, self.d_model, device=images.device))) # 9 zones
                relation_logits_list.append(torch.empty((0, self.relation_head[-1].out_features), device=images.device))
                continue

            # ROI feature extraction (Simplified point sampling for this implementation)
            # Boxes are [x_c, y_c, w, h] normalized or absolute. Assuming absolute up to 224
            # We map the center to the feature map grid (H'=7, W'=7 for resnet18 224)
            feat_h, feat_w = feature_maps.size(2), feature_maps.size(3)
            # Normalize centers to [0, 1] then multiply by feat_dim
            centers_x = (boxes[:, 0] / 224.0) * feat_w
            centers_y = (boxes[:, 1] / 224.0) * feat_h

            centers_x = torch.clamp(centers_x.long(), 0, feat_w - 1)
            centers_y = torch.clamp(centers_y.long(), 0, feat_h - 1)

            # Extract point features (N, d_model)
            obj_feats = feature_maps[i, :, centers_y, centers_x].t()

            # Add class and box encodings
            c_emb = self.class_emb(classes)
            b_emb = self.box_emb(boxes / 224.0)

            # Combine
            tokens = obj_feats + c_emb + b_emb # (N, d_model)

            # Transformer
            encoded_tokens = self.encoder(tokens.unsqueeze(0)).squeeze(0) # (N, d_model)

            # Scene Head (Global Pool)
            global_repr = encoded_tokens.mean(dim=0, keepdim=True) # (1, d_model)
            scene_logits = self.scene_head(global_repr) # (1, num_scenes)
            scene_logits_list.append(scene_logits)

            # Region Head (Assign tokens to 9 zones based on centroid, pool, then classify)
            # Zones are a 3x3 grid
            zone_feats = torch.zeros(9, self.d_model, device=images.device)
            zone_counts = torch.zeros(9, 1, device=images.device)

            grid_x = torch.clamp((boxes[:, 0] / 224.0 * 3).long(), 0, 2)
            grid_y = torch.clamp((boxes[:, 1] / 224.0 * 3).long(), 0, 2)
            zone_idx = grid_y * 3 + grid_x

            for z in range(9):
                mask = (zone_idx == z)
                if mask.any():
                    zone_feats[z] = encoded_tokens[mask].mean(dim=0)

            region_logits = self.region_head(zone_feats) # (9, num_zones)
            region_logits_list.append(region_logits)

            # Relation Head
            if edges.size(1) > 0:
                e_i, e_j = edges[0], edges[1]
                pair_repr = torch.cat([encoded_tokens[e_i], encoded_tokens[e_j]], dim=-1) # (E, d_model*2)
                rel_logits = self.relation_head(pair_repr) # (E, num_predicates)
                relation_logits_list.append(rel_logits)
            else:
                relation_logits_list.append(torch.empty((0, self.relation_head[-1].out_features), device=images.device))

        # Stack scene logits for batch
        scene_logits = torch.cat(scene_logits_list, dim=0)

        return relation_logits_list, region_logits_list, scene_logits
