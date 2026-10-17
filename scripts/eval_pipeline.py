import argparse
import os
import csv
import time
import torch
from hcaf.guards import check_mode, get_out_dir, guard_real_run, stamp_file
from hcaf.l3_transformer import HCAFSemanticTransformer

def run_pipeline(mode):
    out_dir = get_out_dir(mode)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 100% Real parameter counting and latency measurement logic
    model_l3 = HCAFSemanticTransformer(d_model=256, num_heads=8, num_layers=4).to(device)
    model_l3.eval()

    params_l3 = sum(p.numel() for p in model_l3.parameters() if p.requires_grad) / 1e6

    print(f"Running Pipeline Evaluator in {mode.upper()} mode...")

    try:
        from ultralytics import YOLO
        model_l1 = YOLO("yolov8n.pt") # Real load
        params_l1 = sum(p.numel() for p in model_l1.parameters() if p.requires_grad) / 1e6

        # Real L1 Inference
        t0 = time.perf_counter()
        # Mock image tensor to measure pure PyTorch inference speed
        mock_img = torch.randn(1, 3, 640, 640).to(device)
        _ = model_l1(mock_img, verbose=False)
        t_l1 = time.perf_counter()
        l1_latency = (t_l1 - t0) * 1000.0
    except Exception as e:
        print(f"EXPECTED CRASH: YOLO or weights missing. {e}")
        params_l1 = "[REAL_PARAM]"
        l1_latency = "[REAL_LATENCY]"

    # Real L2/L3 Inference Speed test
    t2 = time.perf_counter()
    x_img = torch.randn(1, 3, 224, 224).to(device)
    boxes = [torch.tensor([[112.0, 112.0, 50.0, 50.0]], device=device)]
    classes = [torch.tensor([0], device=device)]
    edges = [torch.empty((2,0), dtype=torch.long, device=device)]

    with torch.no_grad():
        rel, reg, scene = model_l3(x_img, boxes, classes, edges)

    t3 = time.perf_counter()
    l3_latency = (t3 - t2) * 1000.0

    if l1_latency == "[REAL_LATENCY]":
        total_latency = "[REAL_LATENCY]"
        total_params = "[REAL_PARAM]"
    else:
        total_latency = l1_latency + l3_latency
        total_params = params_l1 + params_l3

    data = [
        {"Module": "L1 (YOLOv8n)", "Latency (ms)": l1_latency, "Params (M)": params_l1},
        {"Module": "L3 (Transformer)", "Latency (ms)": l3_latency, "Params (M)": params_l3},
        {"Module": "Total Pipeline", "Latency (ms)": total_latency, "Params (M)": total_params}
    ]

    f1 = os.path.join(out_dir, "table_efficiency.csv")
    with open(f1, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["Module", "Latency (ms)", "Params (M)"])
        writer.writeheader()
        writer.writerows(data)
    stamp_file(f1, mode)

    print(f"Evaluations complete. Results saved to {out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    mode = check_mode(parser)
    guard_real_run(mode)
    run_pipeline(mode)
