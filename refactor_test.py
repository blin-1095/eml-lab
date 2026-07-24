import torch
import gc
from torch.utils.data import DataLoader
import numpy as np

# ---------------------------------------------------------
# 1. IMPORTS
# ---------------------------------------------------------
from pipelines.baseline_pipeline import BaselinePipeline
from pipelines.augmentation_pipeline import AugmentationPipeline
from pipelines.pruning_pipeline import PruningPipeline
from pipelines.person_only_pipeline import PersonOnlyPipeline

from utils.dataloader import VOCDataLoader, VOCDataLoaderPerson
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

# ---------------------------------------------------------
# 2. HARDWARE & HYPERPARAMETERS
# ---------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Standard training hyperparameters
LEARNING_RATE = 1e-5 
TRAIN_BATCH_SIZE = 180
EVAL_BATCH_SIZE = 180  
EPOCHS = 5
NUM_PRUNING_RATIOS = 5
TRANSFORM_PROBABILITY = 0.15

# Person-only pipeline hyperparameters
PERSON_LEARNING_RATE = 0.001
PERSON_EPOCHS = 20
PERSON_BATCH_SIZE = 128

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
test_loader_person = VOCDataLoaderPerson(split="val", batch_size=1)

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
    
    # -----------------------------------------------------
    # Phase 1: Run Baseline Pipeline (Control Group)
    # -----------------------------------------------------
    print("\n" + "="*50)
    print("🚀 STARTING BASELINE PIPELINE (Control)")
    print("="*50)
    
    baseline_pipeline = BaselinePipeline(
        pipeline_name="Baseline Pipeline",
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=DEVICE,
        learning_rate=LEARNING_RATE,
        epochs=EPOCHS,
        sd_path="state_dicts/voc_pretrained.pt"
    )
    
    baseline_results = baseline_pipeline.run()
    
    # Clean up baseline from memory and VRAM
    del baseline_pipeline
    gc.collect()
    torch.cuda.empty_cache()

    # -----------------------------------------------------
    # Phase 2: Run Augmentation Pipeline (Test Group)
    # -----------------------------------------------------
    print("\n" + "="*50)
    print("🚀 STARTING AUGMENTED PIPELINE")
    print("="*50)
    
    augmentation_pipeline = AugmentationPipeline(
        pipeline_name="Augmented Training Pipeline",
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=DEVICE,
        learning_rate=LEARNING_RATE,
        epochs=EPOCHS,
        sd_path="state_dicts/voc_pretrained.pt",
        transform_generator=transform_generator
    )
    
    augmentation_results = augmentation_pipeline.run()
    
    # Clean up augmentation pipeline from memory and VRAM
    del augmentation_pipeline
    gc.collect()
    torch.cuda.empty_cache()

    # -----------------------------------------------------
    # Phase 3: Run Pruning Pipeline
    # -----------------------------------------------------
    pruning_ratios = np.linspace(0, 0.8, NUM_PRUNING_RATIOS)

    baseline_fps = 1.0
    best_score = float('-inf')  # Start at negative infinity to maximize AP score
    best_ratio = None
    results_log = {}

    for ratio in pruning_ratios:
        print(f"\n" + "="*50)
        print(f"🚀 STARTING PRUNING PIPELINE (Ratio: {ratio:.2f})")
        print("="*50)

        pipeline = PruningPipeline(
            pipeline_name=f"Pruned_Pipeline_{int(ratio*100)}",
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            device=DEVICE,
            learning_rate=LEARNING_RATE,
            epochs=EPOCHS,
            sd_path="state_dicts/voc_pretrained.pt",  
            pruning_ratio=ratio
        )
        
        # 1. Run the pipeline and get the unified dictionary
        results = pipeline.run()
        
        # 2. Extract metrics directly from the dictionary
        current_fps = results["fps"]
        test_loss = results["test_loss"]
        current_ap = results["ap"]
        
        # 3. Calculate Speed-Adjusted AP Score
        if ratio == 0.0:
            baseline_fps = current_fps
            current_score = current_ap  
        else:
            speedup_factor = current_fps / baseline_fps if baseline_fps > 0 else 1.0
            current_score = current_ap * speedup_factor 
            
        print(f"\n[*] Ratio {ratio:.2f} -> AP: {current_ap:.4f} | Loss: {test_loss:.4f} | FPS: {current_fps:.1f} | Score: {current_score:.4f}")
        
        results_log[ratio] = current_score
        
        # 4. Track the best pruning ratio (Maximize the score)
        if current_score > best_score:
            best_score = current_score
            best_ratio = ratio
            
        del pipeline
        gc.collect()
        torch.cuda.empty_cache()

    print(f"\n🏆 Best Pruning Ratio: {best_ratio:.2f} (Score: {best_score:.4f})")

    # -----------------------------------------------------
    # Phase 4: Run Person-Only Detection Pipeline
    # -----------------------------------------------------
    print("\n" + "="*50)
    print("🚀 STARTING PERSON-ONLY DETECTION PIPELINE")
    print("="*50)

    person_pipeline = PersonOnlyPipeline(
        pipeline_name="Person Only Detection Pipeline",
        train_loader=train_loader_person,
        val_loader=test_loader_person,
        test_loader=test_loader_person,
        device=DEVICE,
        learning_rate=PERSON_LEARNING_RATE,
        epochs=PERSON_EPOCHS,
        sd_path="state_dicts/voc_pretrained.pt"
    )

    person_results = person_pipeline.run()
    
    # Clean up person-only pipeline from memory and VRAM
    del person_pipeline
    gc.collect()
    torch.cuda.empty_cache()

    print("\n[*] Master script finished successfully!")