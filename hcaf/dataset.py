import torch
from torch.utils.data import Dataset, DataLoader
import json
import os
from PIL import Image
import torchvision.transforms as T
from hcaf.l2_spatial import build_geometric_graph

class HCAFDataset(Dataset):
    def __init__(self, root_dir, split="train", transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform or T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.labels_dir = os.path.join(root_dir, "labels", split)
        self.images_dir = os.path.join(root_dir, "images", split)

        self.samples = []
        if os.path.exists(self.labels_dir):
            for f in os.listdir(self.labels_dir):
                if f.endswith(".json"):
                    self.samples.append(f)

        # Define mappings based on proxy taxonomy
        self.scene_map = {"clean": 0, "minor_litter": 1, "overflow_alert": 2, "illegal_dump_alert": 3, "water_contamination_alert": 4}
        self.zone_map = {"clean": 0, "scattered_litter": 1, "bin_near_capacity": 2, "bin_overflow": 3, "illegal_dump": 4, "water_contamination": 5}
        self.predicate_map = {"none": 0, "in_water": 1, "contained_in": 2, "adjacent_to_overflowing": 3, "far_from_illegal_dump": 4, "on_road": 5, "near": 6}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        label_file = self.samples[idx]
        with open(os.path.join(self.labels_dir, label_file), 'r') as f:
            annotation = json.load(f)

        img_name = label_file.replace(".json", ".jpg")
        img_path = os.path.join(self.images_dir, img_name)

        # 100% Real Image Loading (Will crash if file missing, as intended)
        image = Image.open(img_path).convert("RGB")
        img_tensor = self.transform(image)

        # Real Object Box extraction
        objects = annotation.get("O", [])
        boxes = []
        classes = []
        for obj in objects:
            # Assumes box format: [x_center, y_center, width, height]
            boxes.append(obj["bbox"])
            classes.append(obj["class"])

        # L2 Spatial Graph Construction
        if len(boxes) > 0:
            edge_list, _ = build_geometric_graph(boxes)
            if len(edge_list) > 0:
                edges = torch.tensor(edge_list, dtype=torch.long).t() # Shape: (2, num_edges)
            else:
                edges = torch.empty((2, 0), dtype=torch.long)
        else:
            edges = torch.empty((2, 0), dtype=torch.long)

        # Real Targets parsing
        scene_str = annotation.get("S", "clean")
        scene_target = torch.tensor(self.scene_map.get(scene_str, 0), dtype=torch.long)

        # Zone Targets (Assuming 3x3 grid = 9 zones)
        zone_targets = []
        for z in annotation.get("Z", []):
            zone_targets.append(self.zone_map.get(z["status"], 0))
        # Pad or truncate to 9
        zone_targets = (zone_targets + [0]*9)[:9]
        zone_target_tensor = torch.tensor(zone_targets, dtype=torch.long)

        # Relation Targets (Mapping edges to predicates)
        relations = annotation.get("R", [])
        rel_dict = {(int(r["subject"].split("_")[1]), int(r["object"].split("_")[1])): r["predicate"] for r in relations}

        relation_targets = []
        for i in range(edges.shape[1]):
            src, dst = edges[0, i].item(), edges[1, i].item()
            pred = rel_dict.get((src, dst), "none")
            relation_targets.append(self.predicate_map.get(pred, 0))

        relation_target_tensor = torch.tensor(relation_targets, dtype=torch.long)

        return {
            "image_id": annotation.get("image_id", "unknown"),
            "image": img_tensor,
            "boxes": torch.tensor(boxes, dtype=torch.float32) if len(boxes) > 0 else torch.empty((0, 4)),
            "classes": torch.tensor(classes, dtype=torch.long),
            "edges": edges,
            "scene_target": scene_target,
            "zone_targets": zone_target_tensor,
            "relation_targets": relation_target_tensor
        }

def get_dataloader(root_dir, split="train", batch_size=4, shuffle=True):
    dataset = HCAFDataset(root_dir, split)

    # Custom collate_fn needed because boxes, classes, edges, relations have variable lengths
    def collate_fn(batch):
        return {
            "image_id": [b["image_id"] for b in batch],
            "image": torch.stack([b["image"] for b in batch]),
            "boxes": [b["boxes"] for b in batch],
            "classes": [b["classes"] for b in batch],
            "edges": [b["edges"] for b in batch],
            "scene_target": torch.stack([b["scene_target"] for b in batch]),
            "zone_targets": torch.stack([b["zone_targets"] for b in batch]),
            "relation_targets": [b["relation_targets"] for b in batch]
        }

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
