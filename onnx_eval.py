import time
import torch
import numpy as np
from tqdm import tqdm
import onnxruntime as ort

from utils.metrics import precision_recall_levels, ap
from utils.yolo import filter_boxes, nms
from utils.logger import ExperimentLogger

def evaluate_onnx_model(onnx_path: str, test_loader, pipeline_name: str):
    """
    Benchmarks speed and evaluates AP for an exported ONNX model.
    Saves the results using the standard ExperimentLogger.
    """
    print(f"\n{'='*50}\n EVALUATING ONNX MODEL: {pipeline_name}\n{'='*50}")
    
    # 1. Setup ONNX Session
    session = ort.InferenceSession(onnx_path, providers=['TensorrtExecutionProvider'])
    input_name = session.get_inputs()[0].name
    
    # 2. Benchmark Inference Speed
    print("[*] Benchmarking Inference Speed (CPU)...")
    warmup_tensor, _ = next(iter(test_loader))
    warmup_numpy = warmup_tensor.numpy()
    for _ in range(5):
        session.run(None, {input_name: warmup_numpy})
        
    total_time = 0.0
    total_images = 0
    
    for i, (images, _) in enumerate(test_loader):
        if i >= 50: break
        
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
    print(f"Inference Speed: {fps:.1f} FPS")

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
    print(f"Final AP: {final_ap:.4f}")
    
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