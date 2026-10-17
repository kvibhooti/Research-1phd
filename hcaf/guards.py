import argparse
import os
import sys
import json
import hashlib
import subprocess

def check_mode(parser):
    parser.add_argument('--mode', type=str, choices=['unit', 'smoke', 'proxy', 'final'], required=True,
                        help="Run mode: unit/smoke (synthetic, local), proxy (public data), final (private data)")
    args = parser.parse_args()
    return args.mode

def get_out_dir(mode):
    if mode in ['unit', 'smoke']:
        d = "results/_smoke"
    elif mode == 'proxy':
        d = "results/proxy"
    elif mode == 'final':
        d = "results/final"
    else:
        raise ValueError(f"Unknown mode: {mode}")
    os.makedirs(d, exist_ok=True)
    return d

def guard_real_run(mode):
    if mode in ['unit', 'smoke']:
        return # allowed to bypass checks

    # 1. Manifest check
    if not os.path.exists("DATASET_MANIFEST.json"):
        sys.exit("GUARD FAILURE: DATASET_MANIFEST.json is missing.")

    with open("DATASET_MANIFEST.json", "r") as f:
        manifest = json.load(f)
        if "name" not in manifest or "md5" not in manifest:
             sys.exit("GUARD FAILURE: Invalid DATASET_MANIFEST.json.")

    # 2. Git clean check
    res = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)
    if res.stdout.strip() != "":
         sys.exit("GUARD FAILURE: Working tree is dirty. Commit changes before running a real evaluation.")

def stamp_file(filepath, mode, manifest_path="DATASET_MANIFEST.json"):
    header = ""
    if mode in ['unit', 'smoke']:
        header = "SMOKE/SYNTHETIC: NOT FOR PAPER\n"
    else:
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
        res = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
        git_hash = res.stdout.strip()
        header = f"DATASET: {manifest.get('name', 'UNKNOWN')} | MODE: {mode} | COMMIT: {git_hash}\n"

    try:
        with open(filepath, 'r') as original: data = original.read()
    except FileNotFoundError:
        data = ""

    with open(filepath, 'w') as modified: modified.write(header + data)
