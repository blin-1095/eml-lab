import os
import argparse
import torch

from evaluate import evaluate_model
from utils.dataloader import VOCDataLoader, VOCDataLoaderPerson

def main():
    parser = argparse.ArgumentParser(description="Batch Evaluate Models in state_dict/separate")
    parser.add_argument("--dir", type=str, default="state_dict/separate", help="Directory containing the state dicts")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for evaluation")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[*] Using device: {device}")

    if not os.path.exists(args.dir):
        print(f"[!] Error: Directory '{args.dir}' does not exist.")
        return

    # Find all .pt or .pth files in the target directory
    sd_files = [f for f in os.listdir(args.dir) if f.endswith('.pt') or f.endswith('.pth')]
    
    if not sd_files:
        print(f"[!] No state dicts found in '{args.dir}'.")
        return

    print(f"[*] Found {len(sd_files)} models to evaluate in '{args.dir}'.")

    # Dictionary to cache the dataloaders so we don't re-parse the VOC XMLs for every model
    loaders = {}

    for sd_file in sorted(sd_files):
        sd_path = os.path.join(args.dir, sd_file)
        pipeline_name = os.path.splitext(sd_file)[0]
        
        # Peek at weights to determine the number of classes (1 or 20)
        temp_sd = torch.load(sd_path, map_location=device, weights_only=True)
        
        if 'conv9.weight' not in temp_sd:
            print(f"\n[!] Skipping '{sd_file}', invalid format (missing conv9.weight)")
            continue
            
        classes = int((temp_sd['conv9.weight'].shape[0] / 5) - 5)
        
        # Lazily initialize dataloaders based on the number of classes
        if classes not in loaders:
            print(f"\n[*] Initializing Dataloader for {classes} classes... (This only happens once)")
            if classes == 1:
                loaders[classes] = VOCDataLoaderPerson(split="test", batch_size=args.batch_size)
            else:
                loaders[classes] = VOCDataLoader(split="test", batch_size=args.batch_size)
        
        test_loader = loaders[classes]

        # Run the evaluation using the logic from evaluate.py
        try:
            evaluate_model(
                sd_path=sd_path,
                device=device,
                test_loader=test_loader,
                pipeline_name=pipeline_name,
                export_onnx=False
            )
        except Exception as e:
            print(f"\n[!] Failed to evaluate '{pipeline_name}': {e}")
            
    print(f"\n{'='*50}")
    print(f"[*] ALL DONE! Summaries saved to the raw_data/ directory.")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()