import torch
import gc
import os
from torch.utils.data import DataLoader
import numpy as np
import copy

# ---------------------------------------------------------
# 1. IMPORTS
# ---------------------------------------------------------
from tinyyolov2 import TinyYoloV2

from pipelines.baseline_pipeline import BaselinePipeline
from pipelines.augmentation_pipeline import AugmentationPipeline
from pipelines.pruning_pipeline import PruningPipeline
from pipelines.person_only_pipeline import PersonOnlyPipeline

from utils.dataloader import VOCDataLoader, VOCDataLoaderPerson

from utils.fusion import fuse_sd
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


def export_sd_to_onnx(state_dict: dict, dataloader, device: torch.device, dest_path: str):
    """
    Auto-detects model architecture from a state_dict, initializes a dummy YOLO model, 
    and exports it to ONNX with static shapes for TensorRT.
    """
    channels = [state_dict[f'conv{i}.weight'].shape[0] for i in range(1, 9)]
    num_classes = int((state_dict['conv9.weight'].shape[0] / 5) - 5)
    is_fused = 'bn1.weight' not in state_dict.keys()

    model = TinyYoloV2(num_classes=num_classes, channels=channels, fused=is_fused)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    dummy_input, _ = next(iter(dataloader))
    # slice batch size for Jetson's batch size since axes are now static
    # we only need one image at a time for inference later anyway
    dummy_input = dummy_input[:1].to(device)

    # 4. Ensure the destination folder exists
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)

    torch.onnx.export(
        model, 
        dummy_input, 
        dest_path,
        export_params=True, 
        opset_version=11,
        input_names=['input_image'], 
        output_names=['yolo_output']
    )
    print(f"[*] Successfully exported static ONNX model to '{dest_path}' (Classes: {num_classes})")


# ---------------------------------------------------------
# 2. HARDWARE & HYPERPARAMETERS
# ---------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Standard training hyperparameters
LEARNING_RATE = 1e-5 
TRAIN_BATCH_SIZE = 164
EVAL_BATCH_SIZE = 164 
EPOCHS = 250
NUM_PRUNING_RATIOS = 5
TRANSFORM_PROBABILITY = 0.15

# Person-only pipeline hyperparameters
PERSON_LEARNING_RATE = 0.001
PERSON_BATCH_SIZE = 164

print(f"[*] Initializing master script on device: {DEVICE}")

# ---------------------------------------------------------
# 3. DATASETS & DATALOADERS
# ---------------------------------------------------------
# Standard multi-class datasets
train_loader = VOCDataLoader(split="train", batch_size=TRAIN_BATCH_SIZE)
val_loader = VOCDataLoader(split="val", batch_size=EVAL_BATCH_SIZE)
test_loader = VOCDataLoader(split="test", batch_size=EVAL_BATCH_SIZE)

# Person-only datasets (filtered for class 'person')
train_loader_person = VOCDataLoaderPerson(split="train", batch_size=PERSON_BATCH_SIZE)
val_loader_person = VOCDataLoaderPerson(split="val", batch_size=PERSON_BATCH_SIZE)
test_loader_person = VOCDataLoaderPerson(split="test", batch_size=PERSON_BATCH_SIZE)

# ---------------------------------------------------------
# 4. TRANSFORM GENERATOR
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
# 5. EXECUTION
# ---------------------------------------------------------
if __name__ == "__main__":
    
    print("\n" + "="*50)
    print("EVALUATING BASELINE MODEL (Control - No Training)")
    print("="*50)

    baseline_results = evaluate_model(
        test_loader=test_loader, 
        device=DEVICE,
        pipeline_name="Baseline_Pipeline",
        sd_path="state_dicts/voc_pretrained.pt",
        export_onnx=True
    )

    gc.collect()
    torch.cuda.empty_cache()

    # -----------------------------------------------------
    # Phase 1.5: Additionally export batchnorm-layer-fused inference model
    # -----------------------------------------------------

    print("\n" + "="*50)
    print("FUSING BATCHNORM LAYERS AND EXPORTING")
    print("="*50)

    fused_sd = fuse_sd(state_dict="state_dicts/voc_pretrained.pt", device=DEVICE)

    fused_sd_path = "state_dicts/separate/batchnorm_best_sd.pt"
    torch.save(fused_sd, fused_sd_path)

    baseline_results = evaluate_model(
        test_loader=test_loader, # Make sure this loader matches the num_classes above!
        device=DEVICE,
        pipeline_name="Batchnorm_Pipeline",
        sd_path=fused_sd_path,
    )

    # -----------------------------------------------------
    # Phase 2: Run Augmentation Pipeline (Test Group)
    # -----------------------------------------------------
    print("\n" + "="*50)
    print("STARTING AUGMENTED PIPELINE")
    print("="*50)
    
    augmentation_pipeline = AugmentationPipeline(
        pipeline_name="Augmented Pipeline",
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=DEVICE,
        learning_rate=LEARNING_RATE,
        epochs=EPOCHS,
        transform_generator=transform_generator
    )

    augmentation_pipeline.load_state_dict("state_dicts/voc_pretrained.pt") 
    augmentation_pipeline.run()
    augmentation_pipeline.export_state_dict("state_dicts/separate/augmented_best_sd.pt")
    
    # Clean up augmentation pipeline from memory and VRAM
    del augmentation_pipeline
    gc.collect()
    torch.cuda.empty_cache()

    print("\n[*] Evaluating Augmented Model...")
    evaluate_model(
        sd_path="state_dicts/augmented_best_sd.pt",
        device=DEVICE,              
        test_loader=test_loader,  
        pipeline_name="Augmented Training Pipeline",
    )

    # -----------------------------------------------------
    # Phase 3: Run Pruning Pipeline
    # -----------------------------------------------------
    pruning_ratios = np.linspace(0, 0.8, NUM_PRUNING_RATIOS)

    baseline_fps = 1.0
    best_score = float('-inf')  # Start at negative infinity to maximize AP score
    best_ratio = None
    results_log = {}

    current_sd = torch.load("state_dicts/voc_pretrained.pt", map_location=DEVICE, weights_only=True)
    current_global_sparsity = 0.0

    for target_ratio in pruning_ratios:

        target_ratio_int = int(target_ratio * 100)
        pipeline_name = f"Pruned_Pipeline_{target_ratio_int}"

        if target_ratio == 0.0:
            relative_ratio = 0.0
        else:
            relative_ratio = (target_ratio - current_global_sparsity) / (1.0 - current_global_sparsity)

        print(f"\n" + "="*50)
        print(f"STARTING ITERATIVE PRUNING (Global Target: {target_ratio:.2f} | Relative Drop: {relative_ratio:.2f})")
        print("="*50)

        pipeline = PruningPipeline(
            pipeline_name=pipeline_name,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            device=DEVICE,
            learning_rate=LEARNING_RATE,
            epochs=EPOCHS,
            pruning_ratio=relative_ratio
        )
        
        # 1. Run the pipeline and get the unified dictionary

        saved_sd_path = f"state_dicts/separate/pruned_pipeline_{target_ratio_int}_best_sd.pt"

        pipeline.load_state_dict(current_sd)
        sd = pipeline.run()
        pipeline.export_state_dict(saved_sd_path)

        del pipeline
        gc.collect()
        torch.cuda.empty_cache()

        print(f"\n[*] Evaluating Pruned Model (Ratio {target_ratio:.2f}) to determine efficiency...")
        
        results = evaluate_model(
            test_loader=test_loader, # Make sure this loader matches the num_classes above!
            device=DEVICE,
            pipeline_name=pipeline_name,
            sd_path=saved_sd_path,
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
        current_sd = copy.deepcopy(sd)
            

    print(f"\nBest Iterative Pruning Ratio: {best_ratio:.2f} (Score: {best_score:.4f})")

    # -----------------------------------------------------
    # Phase 4: Run Person-Only Detection Pipeline
    # -----------------------------------------------------
    print("\n" + "="*50)
    print("STARTING PERSON-ONLY DETECTION PIPELINE")
    print("="*50)

    person_pipeline = PersonOnlyPipeline(
        pipeline_name="Person Only Detection Pipeline",
        train_loader=train_loader_person,
        val_loader=val_loader_person,
        test_loader=test_loader_person,
        device=DEVICE,
        learning_rate=PERSON_LEARNING_RATE,
        epochs=EPOCHS,
    )

    person_pipeline.load_state_dict("state_dicts/voc_pretrained.pt") 
    person_pipeline.run()
    person_pipeline.export_state_dict("state_dicts/separate/person_best_sd.pt")
    
    # Clean up person-only pipeline from memory and VRAM
    del person_pipeline
    gc.collect()
    torch.cuda.empty_cache()

    print("\n[*] Evaluating Person-Only Model...")
    evaluate_model(
        sd_path="state_dicts/separate/person_best_sd.pt",
        device=DEVICE,                
        test_loader=test_loader_person,   
        pipeline_name="Person Only Detection Pipeline",
    )

    # 1. Apply Static INT8 Quantization (using QDQ format)
    apply_static_quantization(
        input_onnx_path="models/baseline_pipeline.onnx",
        output_onnx_path="models/baseline_int8_quantized.onnx",
        calib_loader=val_loader
    )

    # 2. Evaluate and Log the Quantized Model
    evaluate_onnx_model(
        onnx_path="state_dists/separate/baseline_int8_quantized.onnx",
        test_loader=test_loader,
        pipeline_name="ONNX_INT8_Quantized"
    )

    print("\n[*] Master script finished successfully!")