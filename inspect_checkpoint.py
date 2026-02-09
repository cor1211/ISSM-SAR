import torch
import sys
import os

def inspect_checkpoint(path):
    if not os.path.exists(path):
        print(f"Error: File not found: {path}")
        return

    print(f"--- Inspecting Checkpoint: {path} ---")
    try:
        checkpoint = torch.load(path, map_location='cpu')
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return

    # Check keys
    keys = checkpoint.keys()
    print(f"Top-level keys: {list(keys)}")

    # Check Epoch and Step
    epoch = checkpoint.get('epoch', 'N/A')
    global_step = checkpoint.get('global_step', 'N/A')
    print(f"Epoch: {epoch}")
    print(f"Global Step: {global_step}")

    # Check PL version
    pl_version = checkpoint.get('pytorch-lightning_version', 'N/A')
    print(f"PyTorch Lightning Version: {pl_version}")

    # Check Callbacks
    callbacks = checkpoint.get('callbacks', {})
    if callbacks:
        print(f"Callbacks present: {list(callbacks.keys())}")
        if 'ModelCheckpoint' in callbacks:
            print(f"ModelCheckpoint state: {callbacks['ModelCheckpoint']}")
    else:
        print("Callbacks: None or Empty")

    # Check State Dict
    state_dict = checkpoint.get('state_dict', {})
    print(f"Model weights count: {len(state_dict)}")
    
    # Check optimizer states
    optimizer_states = checkpoint.get('optimizer_states', [])
    print(f"Optimizer states count: {len(optimizer_states)}")

    # Check loops
    loops = checkpoint.get('loops', {})
    if loops:
        print("Loops state present")
    else:
        print("Loops state key missing (common in older PL versions or manual saves)")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inspect_checkpoint.py <path_to_checkpoint>")
    else:
        inspect_checkpoint(sys.argv[1])
