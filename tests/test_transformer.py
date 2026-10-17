import pytest
import torch
from hcaf.l3_transformer import HCAFSemanticTransformer

def test_model_forward():
    model = HCAFSemanticTransformer(d_model=64, num_heads=2, num_layers=1)

    # 2 images in batch
    images = torch.randn(2, 3, 224, 224)

    # Image 1: 2 objects
    boxes1 = torch.tensor([[100.0, 100.0, 50.0, 50.0], [150.0, 150.0, 50.0, 50.0]])
    classes1 = torch.tensor([0, 1])
    edges1 = torch.tensor([[0], [1]], dtype=torch.long)

    # Image 2: 1 object, no edges
    boxes2 = torch.tensor([[50.0, 50.0, 20.0, 20.0]])
    classes2 = torch.tensor([2])
    edges2 = torch.empty((2, 0), dtype=torch.long)

    boxes_list = [boxes1, boxes2]
    classes_list = [classes1, classes2]
    edges_list = [edges1, edges2]

    rel_logits, reg_logits, scene_logits = model(images, boxes_list, classes_list, edges_list)

    # Assert scene logits shape (batch_size, num_scenes=5 default)
    assert scene_logits.shape == (2, 5)

    # Assert region logits shape: list of (9, num_zones=6 default)
    assert len(reg_logits) == 2
    assert reg_logits[0].shape == (9, 6)

    # Assert relation logits shape: list. First img has 1 edge, second has 0.
    assert len(rel_logits) == 2
    assert rel_logits[0].shape == (1, 7) # default num_predicates=7
    assert rel_logits[1].shape == (0, 7)
