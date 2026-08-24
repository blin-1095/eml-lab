import torch
import gc
from torch.utils.data import DataLoader
import numpy as np
import copy
import os

# ---------------------------------------------------------
# 1. IMPORTS
# ---------------------------------------------------------
from pipelines.augmented_po_pipeline import AugmentedPersonOnlyPipeline
from pipelines.augmented_pruning_pipeline import AugmentedPruningPipeline

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

from tinyyolov2 import TinyYoloV2


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
# HARDWARE & HYPERPARAMETERS
# ---------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Standard training hyperparameters
LEARNING_RATE = 1e-5 
TRAIN_BATCH_SIZE = 180
EVAL_BATCH_SIZE = 180  
EPOCHS = 300
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
    # Run Person Only Pipeline with data augmentation
    # -----------------------------------------------------
    print("\n" + "="*50)
    print("STARTING PERSON ONLY PIPELINE")
    print("="*50)
    
    augmented_po_pipeline = AugmentedPersonOnlyPipeline(
        pipeline_name="Augmented PO Pipeline",
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=DEVICE,
        learning_rate=LEARNING_RATE,
        epochs=EPOCHS,
        transform_generator=transform_generator
    )

    augmented_po_pipeline.load_state_dict("state_dicts/voc_pretrained.pt") 
    augmented_po_sd = augmented_po_pipeline.run()
    augmented_po_pipeline.export_state_dict("state_dicts/master/augmented_po.pt")

    # Export interative result
    export_sd_to_onnx(
        state_dict=augmented_po_sd, 
        dataloader=test_loader, 
        device=DEVICE, 
        dest_path="models/po_model.onnx"
    )

    # Clean up baseline from memory and VRAM
    del augmented_po_pipeline
    gc.collect()
    torch.cuda.empty_cache()
    
    # -----------------------------------------------------
    # Run Pruning Pipeline
    # -----------------------------------------------------
    pruning_ratios = np.linspace(0, 0.8, NUM_PRUNING_RATIOS)

    pruning_sd = augmented_po_sd
    current_global_sparsity = 0.0

    for target_ratio in pruning_ratios:

        target_ratio_int = int(round(target_ratio * 100))
        pipeline_name = f"Pruned_Pipeline_{target_ratio_int}"

        if target_ratio == 0.0:
            continue
        else:
            relative_ratio = (target_ratio - current_global_sparsity) / (1.0 - current_global_sparsity)

        print(f"\n" + "="*50)
        print(f"STARTING ITERATIVE PRUNING (Global Target: {target_ratio:.2f} | Relative Drop: {relative_ratio:.2f})")
        print("="*50)

        pipeline = AugmentedPruningPipeline(
            pipeline_name=pipeline_name,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            device=DEVICE,
            learning_rate=LEARNING_RATE,
            epochs=EPOCHS,
            pruning_ratio=relative_ratio,
            transform_generator=transform_generator
        )
        
        saved_sd_path = f"state_dicts/master/pruned_pipeline_{target_ratio_int}_best_sd.pt"
        saved_model_path = f"models/pruned_pipeline_{target_ratio_int}_model.onnx"

        pipeline.load_state_dict(pruning_sd)
        sd = pipeline.run()
        pipeline.export_state_dict(saved_sd_path)

        export_sd_to_onnx(
            state_dict=sd, 
            dataloader=test_loader, 
            device=DEVICE, 
            dest_path=saved_model_path
        )

        del pipeline
        gc.collect()
        torch.cuda.empty_cache()

        current_global_sparsity = target_ratio
        pruning_sd = copy.deepcopy(sd)
            
            
    # -----------------------------------------------------
    # Export batchnorm-layer-fused inference model
    # -----------------------------------------------------

    print("\n" + "="*50)
    print("FUSING BATCHNORM LAYERS AND EXPORTING")
    print("="*50)

    fused_sd = fuse_sd(state_dict=f"state_dicts/master/pruned_pipeline_20_best_sd.pt", device=DEVICE)
    torch.save(fused_sd, "state_dicts/master/fused.pt")


    export_sd_to_onnx(
        state_dict=fused_sd, 
        dataloader=test_loader, 
        device=DEVICE, 
        dest_path="models/fused_model.onnx"
    )

    #-----------------------------------------------------
    #Apply static INT8 quantization and export final model
    #-----------------------------------------------------

    print("\n" + "="*50)
    print("APPLYING ONNX QUANTIZATION AND EXPORTING")
    print("="*50)

    # dummy dataloader that loads the batch size fitting for the jetson
    onnx_dummy_loader = VOCDataLoaderPerson(split="val", batch_size=1)

    # Apply Static INT8 Quantization (using QDQ format)
    apply_static_quantization(
        input_onnx_path="models/fused_model.onnx",
        output_onnx_path="models/tinyyolov2_improved.onnx",
        calib_loader=onnx_dummy_loader
    )

    # Evaluate and Log the Quantized Model
    evaluate_onnx_model(
        onnx_path="models/tinyyolov2_improved.onnx",
        test_loader=onnx_dummy_loader,
        pipeline_name="All_improv_model"
    )

    print("\n[*] Training script finished successfully!")