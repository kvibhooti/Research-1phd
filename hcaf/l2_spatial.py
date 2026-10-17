import numpy as np

def compute_iou(box1, box2):
    # box format: [xc, yc, w, h]
    x1_min, y1_min = box1[0] - box1[2]/2, box1[1] - box1[3]/2
    x1_max, y1_max = box1[0] + box1[2]/2, box1[1] + box1[3]/2
    x2_min, y2_min = box2[0] - box2[2]/2, box2[1] - box2[3]/2
    x2_max, y2_max = box2[0] + box2[2]/2, box2[1] + box2[3]/2

    inter_xmin = max(x1_min, x2_min)
    inter_ymin = max(y1_min, y2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)

    if inter_xmax <= inter_xmin or inter_ymax <= inter_ymin:
        return 0.0

    inter_area = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)
    box1_area = box1[2] * box1[3]
    box2_area = box2[2] * box2[3]

    return inter_area / (box1_area + box2_area - inter_area)

def build_geometric_graph(boxes):
    n = len(boxes)
    edges = []
    features = []

    for i in range(n):
        for j in range(n):
            if i == j:
                continue

            box_i = boxes[i]
            box_j = boxes[j]

            dist = np.sqrt((box_i[0] - box_j[0])**2 + (box_i[1] - box_j[1])**2)
            angle = np.arctan2(box_j[1] - box_i[1], box_j[0] - box_i[0])
            iou = compute_iou(box_i, box_j)
            containment = min(box_i[2]*box_i[3], box_j[2]*box_j[3]) / max(1e-6, max(box_i[2]*box_i[3], box_j[2]*box_j[3]))

            # Connect based on proximity
            if dist < 0.5 or iou > 0.0:
                edges.append((i, j))
                features.append([dist, angle, iou, containment])

    return edges, features
