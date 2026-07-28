import os
import time
import argparse
import torch
import numpy as np
from tqdm import tqdm
import onnxruntime as ort

from utils.dataloader import VOCDataLoader, VOCDataLoaderPerson
from utils.metrics import precision_recall_levels, ap
from utils.yolo import filter_boxes, nms
from utils.logger import ExperimentLogger

# =====================================================================
# 1. EVALUATION FUNCTION
# =====================================================================

def evaluate_onnx_session(session: ort.InferenceSession, test_loader, pipeline_name: str):
    """
    Benchmarks speed and evaluates AP for an active ONNX InferenceSession.
    Saves the results using the standard ExperimentLogger.
    """
    print(f"\n{'='*50}\n EVALUATING ONNX MODEL: {pipeline_name}\n{'='*50}")
    
    input_name = session.get_inputs()[0].name
    
    # 2. Benchmark Inference Speed
    print("[*] Benchmarking Inference Speed (TensorRT/GPU)...")
    warmup_tensor, _ = next(iter(test_loader))
    warmup_numpy = warmup_tensor.numpy()
    
    # Warm-up phase
    for _ in range(5):
        session.run(None, {input_name: warmup_numpy})
        
    total_time = 0.0
    total_images = 0
    
    for i, (images, _) in enumerate(test_loader):
        if i >= 50: 
            break
        
        images_np = images.numpy()
        batch_size = images_np.shape[0]
        
        # Safety check: skip partial batches to prevent ONNX shape mismatches
        if images_np.shape[0] != test_loader.batch_size:
            continue
            
        start_time = time.time()
        session.run(None, {input_name: images_np})
        total_time += (time.time() - start_time)
        total_images += batch_size
        
    fps = total_images / total_time
    print(f"[*] Inference Speed: {fps:.1f} FPS")

    # 3. Evaluate Average Precision (AP)
    test_precision = []
    test_recall = []
    
    loop = tqdm(test_loader, leave=True, desc="Evaluating AP")
    for images, targets in loop:
        images_np = images.numpy()
        
        # ONNX Inference (Returns a list, we want the first output)
        onnx_outputs = session.run(None, {input_name: images_np})[0]
        
        # Convert Numpy array back to PyTorch tensor for your YOLO post-processing
        outputs = torch.tensor(onnx_outputs)
        
        outputs = filter_boxes(outputs, 0.25)
        outputs = nms(outputs, 0.5)
        
        for i in range(images.size(0)):
            prec, rec = precision_recall_levels(targets[i], outputs[i])
            test_precision.append(prec)
            test_recall.append(rec)
            
    final_ap = ap(test_precision, test_recall)
    print(f"[*] Final AP: {final_ap:.4f}")
    
    # 4. Log Results
    logger = ExperimentLogger(pipeline_name)
    logger.train_losses = []
    logger.val_losses = []
    
    final_results = {
        "test_loss": 0.0,
        "ap": float(final_ap),
        "fps": fps,
        "inference_time_ms": 1000.0 / fps if fps > 0 else 0.0,
        "total_parameters": 0, # Not applicable for raw ONNX files
        "test_precision_levels": test_precision,
        "test_recall_levels": test_recall,
        "pruning_ratio": 0.0 
    }
    
    logger.log_final_metrics(final_results)
    logger.save_report()
    
    return final_results


# =====================================================================
# 2. BATCH RUNNER
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="Batch Evaluate ONNX Models on Jetson TensorRT")
    parser.add_argument("--dir", type=str, default="models", help="Directory containing the .onnx files")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for evaluation")
    args = parser.parse_args()

    if not os.path.exists(args.dir):
        print(f"[!] Error: Directory '{args.dir}' does not exist.")
        return

    onnx_files = [f for f in os.listdir(args.dir) if f.endswith('.onnx')]
    
    if not onnx_files:
        print(f"[!] No .onnx files found in '{args.dir}'.")
        return

    print(f"[*] Found {len(onnx_files)} models to evaluate in '{args.dir}'.")

    # Define execution providers (TensorRT is strictly prioritized for Jetson)
    providers = ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
    loaders = {}

    for onnx_file in sorted(onnx_files):
        onnx_path = os.path.join(args.dir, onnx_file)
        # Append an '_onnx' suffix so the JSON log doesn't overwrite the PyTorch JSON log
        pipeline_name = os.path.splitext(onnx_file)[0] + "_onnx"
        
        print(f"\n[*] Initializing TensorRT Engine for {onnx_file} (This may take a minute...)")
        
        try:
            # 1. Load the ONNX Session
            session = ort.InferenceSession(onnx_path, providers=providers)
            
            # 2. Dynamically determine classes based on ONNX output shape
            # Typical shape: [batch_size, channels, H, W] -> channels = 5 * (5 + classes)
            output_shape = session.get_outputs()[0].shape
            channels = output_shape[1]
            classes = int((channels / 5) - 5)
            
            # 3. Lazily initialize the correct dataloader
            if classes not in loaders:
                print(f"[*] Initializing Dataloader for {classes} classes... (Only happens once)")
                if classes == 1:
                    loaders[classes] = VOCDataLoaderPerson(split="test", batch_size=args.batch_size)
                else:
                    loaders[classes] = VOCDataLoader(split="test", batch_size=args.batch_size)
            
            test_loader = loaders[classes]
            
            # 4. Evaluate using the active session
            evaluate_onnx_session(session, test_loader, pipeline_name)
            
        except Exception as e:
            print(f"\n[!] Failed to evaluate '{onnx_file}': {e}")
            
    print(f"\n{'='*50}")
    print(f"[*] ALL DONE! ONNX summaries saved to the raw_data/ directory.")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()