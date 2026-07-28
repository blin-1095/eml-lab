import os
import time
import copy
import argparse
import torch
import gc
from tqdm import tqdm

from tinyyolov2 import TinyYoloV2
from utils.dataloader import VOCDataLoader, VOCDataLoaderPerson
from utils.logger import ExperimentLogger

from utils.metrics import precision_recall_levels, ap
from utils.yolo import filter_boxes, nms
from utils.loss import YoloLoss

# make pyright shut up
from torch.utils.data import DataLoader
from typing_extensions import Optional


# =====================================================================
# 1. UTILITY FUNCTIONS
# =====================================================================

def calculate_metrics(model: torch.nn.Module, loader, device: torch.device, conf_thresh: float = 0.25, nms_thresh: float = 0.5) -> dict:
    """Runs inference over a dataloader to calculate Test Loss, Precision, Recall, and AP in one pass."""
    model.eval()
    
    # Initialize the loss function exactly as it was in your training pipeline
    criterion = YoloLoss(anchors=model.anchors)
    criterion.seen = 999999  
    
    test_precision = []
    test_recall = []
    total_loss = 0.0
    
    loop = tqdm(loader, leave=False, desc="Evaluating Metrics")
    with torch.no_grad():
        for images, targets in loop:
            images, targets = images.to(device), targets.to(device)
            
            # 1. Compute Test Loss (Requires raw model output)
            raw_outputs = model(images, yolo=False)
            loss_output = criterion(raw_outputs, targets)
            if isinstance(loss_output, tuple):
                loss = loss_output[0]
            else:
                loss = loss_output
                
            total_loss += loss.item()
            
            # 2. Compute AP Metrics (Requires YOLO decoded output)
            outputs = model(images, yolo=True)
            
            # Post-processing: confidence thresholding and NMS
            outputs = filter_boxes(outputs, conf_thresh)
            outputs = nms(outputs, nms_thresh)
            
            # Calculate metrics per image in the batch
            for i in range(images.size(0)):
                prec, rec = precision_recall_levels(targets[i], outputs[i])
                test_precision.append(prec)
                test_recall.append(rec)
            
    # Calculate final 11-point AP and Average Loss
    final_ap = ap(test_precision, test_recall)
    avg_loss = total_loss / len(loader)
    
    return {
        "test_loss": float(avg_loss),
        "ap": float(final_ap),
        "precision_levels": test_precision,
        "recall_levels": test_recall
    }


def benchmark_inference_speed(model: torch.nn.Module, test_loader, device: torch.device, num_batches: int = 50) -> float:
    """Runs a quick inference benchmark on the test set to calculate FPS."""
    model.eval()
    
    # Warm-up phase
    warmup_input, _ = next(iter(test_loader))
    warmup_input = warmup_input.to(device)
    with torch.no_grad():
        for _ in range(10):
            model(warmup_input, yolo=True)
            
    # Benchmark phase
    total_time = 0.0
    total_images = 0
    
    with torch.no_grad():
        for i, (images, _) in enumerate(test_loader):
            if i >= num_batches:
                break
                
            images = images.to(device)
            batch_size = images.shape[0]
            
            if device.type == "cuda":
                torch.cuda.synchronize()
            
            start_time = time.time()
            model(images, yolo=True)
            
            if device.type == "cuda":
                torch.cuda.synchronize()
            
            total_time += (time.time() - start_time)
            total_images += batch_size
            
    fps = total_images / total_time
    return fps


def export_model_onnx(state_dict: dict, test_loader, device: torch.device, dest_path: str):
    """Export the trained model in ONNX format. Auto-detects architecture."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    dest_dir = os.path.dirname(dest_path)
    if dest_dir:
        os.makedirs(dest_dir, exist_ok=True)

    sd = copy.deepcopy(state_dict)

    # Export best model in onnx format
    new_channels = [sd[f'conv{i}.weight'].shape[0] for i in range(1, 9)]
    conv9_filters = sd['conv9.weight'].shape[0]
    num_classes = int((conv9_filters / 5) - 5)
    is_fused = 'bn1.weight' not in sd.keys()

    best_net = TinyYoloV2(num_classes=num_classes, channels=new_channels, fused=is_fused)
    best_net.load_state_dict(state_dict, strict=False)
    best_net.to(device)
    best_net.eval()

    export_input, _ = next(iter(test_loader))
    export_input = export_input[:1].to(device)

    torch.onnx.export(
        best_net,
        export_input,
        dest_path,
        export_params=True,
        opset_version=11,
        input_names=['input_image'],
        output_names=['yolo_output'],
    )

    print(f"[*] Exported model to '{dest_path}'! (Classes: {num_classes}, Fused: {is_fused})")


# =====================================================================
# 2. MAIN EVALUATION LOGIC
# =====================================================================

def evaluate_model(sd_path: str, device: torch.device = None, test_loader: DataLoader = None, pipeline_name: str = None, export_onnx: bool = False):
    """Loads a model, evaluates Test Loss/AP, benchmarks FPS, and logs the results."""
    
    if not os.path.exists(sd_path):
        raise FileNotFoundError(f"Could not find weights at {sd_path}")

    # 1. Load the state dict and AUTO-DETECT ARCHITECTURE
    sd = torch.load(sd_path, map_location=device, weights_only=True)
    
    channels = [sd[f'conv{i}.weight'].shape[0] for i in range(1, 9)]
    num_classes = int((sd['conv9.weight'].shape[0] / 5) - 5)
    is_fused = 'bn1.weight' not in sd.keys()

    if pipeline_name is None:
        raise ValueError("No state dict path passed.")

    if test_loader is None:
        raise ValueError("No dataloader passed.")

    if device is None:
        raise ValueError("No device passed.")
    
        
    print(f"\n{'='*50}")
    print(f"EVALUATING MODEL: {pipeline_name}")
    print(f"{'='*50}")
    print(f"[*] Auto-detected -> Classes: {num_classes} | Fused: {is_fused}")
    print(f"[*] Auto-detected channels: {channels}")
    
    # 3. Instantiate Model
    model = TinyYoloV2(num_classes=num_classes, channels=channels, fused=is_fused)
    model.load_state_dict(sd, strict=False)
    model.to(device)
    model.eval()
    
    # 4. Run your precise Benchmark function
    print("\n[*] Benchmarking Inference Speed...")
    fps = benchmark_inference_speed(model, test_loader, device)
    print(f"Speed: {fps:.1f} FPS")

    # 5. Calculate Test Loss and Average Precision
    print("\n[*] Calculating Test Loss and Average Precision (AP)...")
    metrics_dict = calculate_metrics(model, test_loader, device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Accuracy: {metrics_dict['ap']:.4f} AP | Test Loss: {metrics_dict['test_loss']:.4f} | Parameters: {total_params:,}")
    
    # 6. LOG AND SAVE RAW DATA
    print(f"\n[*] Updating JSON report for '{pipeline_name}'...")
    logger = ExperimentLogger(pipeline_name)
    
    final_results = {
        "test_loss": metrics_dict["test_loss"], 
        "ap": metrics_dict["ap"],
        "fps": fps,
        "inference_time_ms": 1000.0 / fps if fps > 0 else 0.0,
        "total_parameters": total_params,
        "test_precision_levels": metrics_dict["precision_levels"],
        "test_recall_levels": metrics_dict["recall_levels"],
        "pruning_ratio": 0.0 
    }
    
    logger.log_final_metrics(final_results)
    logger.save_report()
    
    # 7. Optional ONNX Export
    if export_onnx:
        onnx_dest = f"models/{pipeline_name.lower()}.onnx"
        print(f"\n[*] Exporting to ONNX...")
        export_model_onnx(sd, test_loader, device, onnx_dest)
    
    # 8. Memory Cleanup
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    print(f"\nDone! Results saved to raw_data/{pipeline_name}.json")
    return final_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="General PyTorch Model Evaluation Script")
    parser.add_argument("--weights", type=str, required=True, help="Path to the .pt state_dict file")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for evaluation")
    parser.add_argument("--name", type=str, default=None, help="Name for the logger output")
    parser.add_argument("--export_onnx", action="store_true", help="Export the model to ONNX")
    
    args = parser.parse_args()
    
    # --- FIX: Create device and dataloader for CLI execution ---
    CLI_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Peek at the weights to know which dataset to load
    temp_sd = torch.load(args.weights, map_location=CLI_DEVICE, weights_only=True)
    classes = int((temp_sd['conv9.weight'].shape[0] / 5) - 5)
    
    if classes == 1:
        cli_loader = VOCDataLoaderPerson(split="test", batch_size=args.batch_size)
    else:
        cli_loader = VOCDataLoader(split="test", batch_size=args.batch_size)
    # -----------------------------------------------------------

    evaluate_model(
        sd_path=args.weights, 
        batch_size=args.batch_size, 
        device=CLI_DEVICE, 
        test_loader=cli_loader, 
        pipeline_name=args.name, 
        export_onnx=args.export_onnx
    )