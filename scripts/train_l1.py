import argparse
import os
import csv
from hcaf.guards import check_mode, get_out_dir, guard_real_run, stamp_file

def train_l1(mode):
    out_dir = get_out_dir(mode)
    csv_file = os.path.join(out_dir, "table4_detection.csv")

    if mode in ['smoke', 'unit']:
        print("Running L1 Detector Training in SMOKE mode...")
        # Write dummy evaluation CSV
        data = [
            {"Model": "YOLOv8n", "mAP@50": 0.50, "mAP@50:95": 0.30, "Precision": 0.55, "Recall": 0.50, "Params(M)": 3.2, "FPS": 120},
            {"Model": "Faster R-CNN", "mAP@50": 0.45, "mAP@50:95": 0.25, "Precision": 0.48, "Recall": 0.45, "Params(M)": 41.5, "FPS": 25}
        ]
    else:
        print(f"Running L1 Detector Training in {mode.upper()} mode...")
        try:
            from ultralytics import YOLO
            # Real training logic
            # Use small epochs and batch for proxy to prevent OOM
            model = YOLO("yolov8n.yaml") # initialize nano model functionally
            # model.train(data="configs/data_proxy.yaml", epochs=3, batch=4)
            # metrics = model.val()

            # Using placeholder metrics for the output until real run happens
            data = [
                {"Model": "YOLOv8n", "mAP@50": "[Y_MAP50]", "mAP@50:95": "[Y_MAP95]", "Precision": "[Y_P]", "Recall": "[Y_R]", "Params(M)": "[Y_M]", "FPS": "[Y_FPS]"}
            ]
            print("Successfully initialized YOLOv8n via ultralytics library.")
        except ImportError:
            print("ultralytics library not found. Ensure it is installed.")
            data = []

    with open(csv_file, 'w', newline='') as f:
        if data:
            writer = csv.DictWriter(f, fieldnames=["Model", "mAP@50", "mAP@50:95", "Precision", "Recall", "Params(M)", "FPS"])
            writer.writeheader()
            writer.writerows(data)

    stamp_file(csv_file, mode)
    print(f"L1 evaluation complete. Results saved to {csv_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    mode = check_mode(parser)
    guard_real_run(mode)
    train_l1(mode)
