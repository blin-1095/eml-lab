import torch
import gc
from torch.utils.data import DataLoader
import numpy as np

# ---------------------------------------------------------
# 1. IMPORTS
# ---------------------------------------------------------
from pipelines.master_pipeline import MasterPipeline
from pipelines.augmentation_pipeline import AugmentationPipeline
from pipelines.pruning_pipeline import PruningPipeline
from pipelines.person_only_pipeline import PersonOnlyPipeline

from utils.dataloader import VOCDataLoader, VOCDataLoaderPerson

from utils.fusion import load_fused_model, export_fused_onnx
from utils.onnx_quantization import apply_static_quantization
from transformation.transform_generator import TransformGenerator
from transformation.horizontal_flip_transform import HorizontalFlipTransform
from transformation.rotation_transform import RotationTransform
from transformation.crop_transform import CropTransform
from transformation.cutout_transform import CutoutTransform
from transformation.color_distortion_transform import ColorDistortionTransform
from transformation.gaussian_noise_transform import GaussianNoiseTransform
from transformation.gaussian_blur_transform import GaussianBlurTransform
from transformation.grayscale_transform import GrayscaleTransform
from transformation.sobel_filter_transform import SobelFilterTransform

from onnx_eval import evaluate_onnx_model
from evaluate import evaluate_model

# ---------------------------------------------------------
# HARDWARE & HYPERPARAMETERS
# ---------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Standard training hyperparameters
LEARNING_RATE = 1e-5 
TRAIN_BATCH_SIZE = 180
EVAL_BATCH_SIZE = 180  
EPOCHS = 3
NUM_PRUNING_RATIOS = 5
TRANSFORM_PROBABILITY = 0.15


print(f"[*] Initializing training script on device: {DEVICE}")

# ---------------------------------------------------------
# DATASETS & DATALOADERS
# ---------------------------------------------------------
# Person-only datasets (filtered for class 'person')
train_loader = VOCDataLoaderPerson(split="train", batch_size=TRAIN_BATCH_SIZE)
val_loader = VOCDataLoaderPerson(split="val", batch_size=EVAL_BATCH_SIZE)
test_loader = VOCDataLoaderPerson(split="val", batch_size=EVAL_BATCH_SIZE)

# ---------------------------------------------------------
# TRANSFORM GENERATOR
# ---------------------------------------------------------
transform_list = [
    HorizontalFlipTransform(),
    RotationTransform(),
    CropTransform(),
    CutoutTransform(),
    ColorDistortionTransform(),
    GaussianNoiseTransform(),
    GaussianBlurTransform(),
    GrayscaleTransform(),
    SobelFilterTransform()
]

transform_generator = TransformGenerator(transformations=transform_list, probability=TRANSFORM_PROBABILITY)
print(f"Loaded {len(transform_list)} transforms.")

# ---------------------------------------------------------
# EXECUTION
# ---------------------------------------------------------
if __name__ == "__main__":
    
    # -----------------------------------------------------
    # Run Master Pipeline
    # Trains for person-only detection with data augmentation
    # -----------------------------------------------------
    print("\n" + "="*50)
    print("STARTING MASTER PIPELINE")
    print("="*50)
    
    master_pipeline = MasterPipeline(
        pipeline_name="Master Pipeline",
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=DEVICE,
        learning_rate=LEARNING_RATE,
        epochs=EPOCHS,
        sd_path="state_dicts/voc_pretrained.pt",
        transform_generator=transform_generator
    )
    
    master_sd = master_pipeline.run()

    # Clean up baseline from memory and VRAM
    del master_pipeline
    gc.collect()
    torch.cuda.empty_cache()
    
    # -----------------------------------------------------
    # Run Pruning Pipeline
    # Test different pruning ratios and decide on the optimal based on Efficency Score
    # -----------------------------------------------------
    pruning_ratios = np.linspace(0, 0.8, NUM_PRUNING_RATIOS)

    baseline_fps = 1.0
    best_score = float('-inf')  # Start at negative infinity to maximize AP score
    best_ratio = None
    results_log = {}

    current_sd_path = "state_dicts/master_pipeline_best_sd.pt"
    current_global_sparsity = 0.0

    for target_ratio in pruning_ratios:

        target_ratio_int = int(target_ratio * 100)
        pipeline_name = f"Pruned_Pipeline_{target_ratio_int}"

        if target_ratio == 0.0:
            relative_ratio = 0.0
        else:
            relative_ratio = (target_ratio - current_global_sparsity) / (1.0 - current_global_sparsity)

        print(f"\n" + "="*50)
        print(f"🚀 STARTING ITERATIVE PRUNING (Global Target: {target_ratio:.2f} | Relative Drop: {relative_ratio:.2f})")
        print(f"[*] Starting weights loaded from: {current_sd_path}")
        print("="*50)

        pipeline = PruningPipeline(
            pipeline_name=pipeline_name,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            device=DEVICE,
            learning_rate=LEARNING_RATE,
            epochs=EPOCHS,
            sd_path=current_sd_path,  
            pruning_ratio=relative_ratio
        )
        
        # 1. Run the pipeline and get the unified dictionary
        pipeline.run()

        del pipeline
        gc.collect()
        torch.cuda.empty_cache()

        saved_sd_path = f"state_dicts/pruned_pipeline_{target_ratio_int}_best_sd.pt"
        print(f"\n[*] Evaluating Pruned Model (Ratio {target_ratio:.2f}) to determine efficiency...")
        
        results = evaluate_model(
            test_loader=test_loader, # Make sure this loader matches the num_classes above!
            device=DEVICE,
            pipeline_name="Baseline_Pipeline",
            sd_path=saved_sd_path
        )

        current_fps = results["fps"]
        test_loss = results["test_loss"]
        current_ap = results["ap"]
        
        # 3. Calculate Speed-Adjusted AP Score
        if target_ratio == 0.0:
            baseline_fps = current_fps
            current_score = current_ap  
        else:
            speedup_factor = current_fps / baseline_fps if baseline_fps > 0 else 1.0
            current_score = current_ap * speedup_factor 
            
        print(f"\n[*] Global Ratio {target_ratio:.2f} -> AP: {current_ap:.4f} | Loss: {test_loss:.4f} | FPS: {current_fps:.1f} | Score: {current_score:.4f}")
        
        results_log[target_ratio] = current_score
        
        # 4. Track the best pruning ratio (Maximize the score)
        if current_score > best_score:
            best_score = current_score
            best_ratio = target_ratio
            best_ratio_int = target_ratio_int
        
        current_global_sparsity = target_ratio
        current_sd_path = saved_sd_path
            

    print(f"\nBest Iterative Pruning Ratio: {best_ratio:.2f} (Score: {best_score:.4f})")

    # -----------------------------------------------------
    # Export batchnorm-layer-fused inference model
    # -----------------------------------------------------

    print("\n" + "="*50)
    print("FUSING BATCHNORM LAYERS AND EXPORTING")
    print("="*50)

    fused_model = load_fused_model(sd_path=f"state_dicts/pruned_pipeline_{int(best_ratio*100)}_best_sd.pt", device=DEVICE)

    warmup, _ = next(iter(test_loader))
    export_fused_onnx(
        model=fused_model, 
        dummy_input=warmup[:1].to(DEVICE), 
        dest_path="models/fused_model.onnx"
    )

    # -----------------------------------------------------
    # Apply static INT8 quantization and export final model
    # -----------------------------------------------------

    # Apply Static INT8 Quantization (using QDQ format)
    apply_static_quantization(
        input_onnx_path="models/fused_model.onnx",
        output_onnx_path="models/baseline_int8_quantized.onnx",
        calib_loader=val_loader
    )

    # Evaluate and Log the Quantized Model
    evaluate_onnx_model(
        onnx_path="models/baseline_int8_quantized.onnx",
        test_loader=test_loader,
        pipeline_name="ONNX_INT8_Quantized"
    )

    print("\n[*] Training script finished successfully!")