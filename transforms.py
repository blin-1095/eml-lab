import torch
import gc
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt

from utils.dataloader import VOCDataLoader 
from tinyyolov2 import TinyYoloV2         
from utils.loss import YoloLoss            
from utils.early_stopping import EarlyStopping
from utils.viz import display_result

from transformation.transform_generator import TransformGenerator
from transformation.horizontal_flip_transform import HorizontalFlipTransform
from transformation.rotation_transform import RotationTransform
from transformation.crop_transform import CropTransform
from transformation.cutout_transform import CutoutTransform
from transformation.color_distortion_transform import ColorDistortionTransform
from transformation.gaussian_noise_transform import GaussianNoiseTransform
from transformation.grayscale_transform import GrayscaleTransform
from transformation.gaussian_blur_transform import GaussianBlurTransform
from transformation.sobel_filter_transform import SobelFilterTransform

# --- CONFIGURATION ---
LEARNING_RATE = 1e-6
TRAIN_BATCH_SIZE = 180
EVAL_BATCH_SIZE = 180  # Can often be scaled up (e.g., 384) to speed up eval
EPOCHS = 50
TRANSFORM_PROBABILITY = 0.3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_epoch(model, loader, optimizer, criterion, transform_generator=None, total_params=0):
    model.train()
    running_loss = 0.0

    loop = tqdm(loader, leave=True, desc="Training")
    for batch_idx, (images, targets) in enumerate(loop):
        images = images.to(DEVICE)
        targets = targets.to(DEVICE)

        # Only apply transformations if a generator was provided
        if transform_generator is not None and total_params > 0:
            batch_size = images.shape[0]
            random_params = torch.rand((batch_size, total_params), device=DEVICE)
            images, targets = transform_generator.transform(random_params, images=images, targets=targets)

        predictions = model(images, yolo=False)
        loss, _ = criterion(predictions, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    return running_loss / len(loader)

def validate_epoch(model, loader, criterion, split_name="Validation"):
    model.eval()
    running_loss = 0.0

    loop = tqdm(loader, leave=True, desc=split_name)
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(loop):
            images = images.to(DEVICE)
            targets = targets.to(DEVICE)

            predictions = model(images, yolo=False)
            loss, _ = criterion(predictions, targets)

            running_loss += loss.item()
            loop.set_postfix(loss=loss.item())

    return running_loss / len(loader)

def run_training_pipeline(pipeline_name, train_loader, val_loader, test_loader, transform_generator=None, total_params=0):
    """Encapsulates the entire training process to ensure a fresh start each time."""
    
    print(f"\n{'='*50}")
    print(f"STARTING PIPELINE: {pipeline_name}")
    print(f"{'='*50}")

    model = TinyYoloV2(num_classes=20)
    state_dict = torch.load('state_dicts/voc_pretrained.pt')
    model.load_state_dict(state_dict)
    model = model.to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion_train = YoloLoss(anchors=model.anchors)
    criterion_eval = YoloLoss(anchors=model.anchors)
    criterion_eval.seen = 999999

    # --- INITIALIZE EARLY STOPPING ---
    # Give it a patience of 15-20 epochs (standard for augmented pipelines)
    early_stopper = EarlyStopping(patience=20, min_delta=0.01)
    
    # Create a safe filename for this specific pipeline's best weights
    clean_name = pipeline_name.split()[0].lower() # e.g., 'baseline' or 'augmented'
    best_model_path = f"state_dicts/transform_best_{clean_name}_model.pt"

    train_losses = []
    val_losses = []

    for epoch in range(EPOCHS):
        print(f"\n--- {pipeline_name} | Epoch [{epoch+1}/{EPOCHS}] ---")
        
        train_loss = train_epoch(model, train_loader, optimizer, criterion_train, transform_generator, total_params)
        print(f"Train Loss: {train_loss:.4f}")
        train_losses.append(train_loss)
        
        val_loss = validate_epoch(model, val_loader, criterion_eval, split_name="Validation")
        print(f"Val Loss: {val_loss:.4f}")
        val_losses.append(val_loss)

        # --- CHECK EARLY STOPPING ---
        early_stopper(val_loss, model, best_model_path)
        
        if early_stopper.early_stop:
            print(f"\n[!] Early stopping triggered at epoch {epoch+1}. No improvement for {early_stopper.patience} epochs.")
            break # Break out of the epoch loop

    # --- LOAD BEST WEIGHTS FOR TESTING ---
    # If early stopping triggered, the current model in memory is overfitted.
    # We must load the best weights we saved before running the final test.
    print(f"\nLoading best weights from {best_model_path} for final test...")
    model.load_state_dict(torch.load(best_model_path))

    print(f"\n--- {pipeline_name} | Final Testing Phase ---")
    test_loss = validate_epoch(model, test_loader, criterion_eval, split_name="Testing")
    print(f"Final Test Loss: {test_loss:.4f}")

    # Optionally truncate the losses lists so the plot only shows up to the stop point
    return train_losses, val_losses, test_loss

def plot_comparative_losses(base_train, base_val, aug_train, aug_val):
    """Plots baseline (no-transforms) vs augmented loss curves."""
    plt.figure(figsize=(12, 7))
    epochs = range(1, len(base_train) + 1)
    
    # Plot Baseline (Dashed Lines)
    plt.plot(epochs, base_train, label='Train Loss (No Transforms)', linestyle='--', marker='o', color='blue', alpha=0.5)
    plt.plot(epochs, base_val, label='Val Loss (No Transforms)', linestyle='--', marker='s', color='orange', alpha=0.5)
    
    # Plot Augmented (Solid Lines)
    plt.plot(epochs, aug_train, label='Train Loss (Augmented)', linestyle='-', marker='o', color='blue', linewidth=2)
    plt.plot(epochs, aug_val, label='Val Loss (Augmented)', linestyle='-', marker='s', color='orange', linewidth=2)
    
    plt.title('Training & Validation Loss: Baseline vs. Augmented', fontsize=15)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.xticks(epochs)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('baseline_vs_augmented_comparison.png')
    print("\nSaved comparative training curves to 'baseline_vs_augmented_comparison.png'")
    plt.show()

def save_debug_images(dataloader, transform_generator, total_params, device, num_images=3, file_prefix='debug'):
    """
    Extracts a batch from the dataloader, applies transformations, 
    and saves multiple images (original vs. transformed) to disk.
    """
    print(f"Exporting {num_images} debug image pairs to disk...")

    # 1. Fetch a single batch from the dataloader
    debug_images, debug_targets = next(iter(dataloader))
    
    # Safety check: ensure we don't request more images than the batch actually holds
    actual_num_images = min(num_images, debug_images.shape[0])

    # Move to GPU if necessary
    debug_images = debug_images.to(device)
    debug_targets = debug_targets.to(device)

    # 2. Generate random parameters for the generator
    debug_params = torch.rand(debug_images.shape[0], total_params, device=device)

    # 3. Apply the transformations
    transformed_images, transformed_targets = transform_generator.transform(
        debug_params, debug_images.clone(), debug_targets.clone()
    )

    # 4. Loop through and save the requested number of images
    for i in range(actual_num_images):
        
        # Extract the i-th image/target and add a dummy batch dimension 
        # so display_result's hardcoded [0,:] indexing continues to work
        orig_img = debug_images[i].unsqueeze(0).cpu()
        orig_tgt = debug_targets[i].unsqueeze(0).cpu()
        
        trans_img = transformed_images[i].unsqueeze(0).cpu()
        trans_tgt = transformed_targets[i].unsqueeze(0).cpu()

        # Save Original
        display_result(
            image=orig_img,  
            output=[],                 
            target=orig_tgt, 
            file_path=f'results/transforms/{file_prefix}_{i}_original.png'
        )

        # Save Transformed
        display_result(
            image=trans_img, 
            output=[], 
            target=trans_tgt, 
            file_path=f'results/transforms/{file_prefix}_{i}_transformed.png'
        )

    print(f"Saved {actual_num_images} pairs of images!")


def main():
    print(f"--- Preparing Environment on {DEVICE} ---")

    train_loader = VOCDataLoader(split="train", batch_size=TRAIN_BATCH_SIZE)
    val_loader = VOCDataLoader(split="val", batch_size=EVAL_BATCH_SIZE)
    test_loader = VOCDataLoader(split="test", batch_size=EVAL_BATCH_SIZE)

    # 1. SETUP TRANSFORMS
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
    total_params = sum([t.get_required_params_amount() for t in transform_list])
    transform_generator = TransformGenerator(transformations=transform_list, probability=TRANSFORM_PROBABILITY)

    print(f"Loaded {len(transform_list)} transforms requiring {total_params} parameters.")

    save_debug_images(
        dataloader=train_loader, 
        transform_generator=transform_generator, 
        total_params=total_params, 
        device=DEVICE,
        num_images=5,
        file_prefix='batch1'
    )

    # 2. RUN BASELINE PIPELINE (No Transforms)
    # By passing transform_generator=None, the train_epoch function will skip augmentations completely.
    base_train, base_val, base_test = run_training_pipeline(
        pipeline_name="Baseline Pipeline (No Augmentations)", 
        train_loader=train_loader, 
        val_loader=val_loader, 
        test_loader=test_loader,
        transform_generator=None,
        total_params=0
    )

    print("\nFlushing RAM and VRAM before Augmented run...")
    gc.collect()                 # Forces Python to delete unreferenced variables
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 3. RUN AUGMENTED PIPELINE (With Transforms)
    # This resets the model to the original state_dict and trains with heavy augmentations.
    aug_train, aug_val, aug_test = run_training_pipeline(
        pipeline_name="Augmented Pipeline (Full Transform Pipeline)", 
        train_loader=train_loader, 
        val_loader=val_loader, 
        test_loader=test_loader,
        transform_generator=transform_generator,
        total_params=total_params
    )

    # 4. FINAL SUMMARY AND PLOTTING
    print("\n" + "="*50)
    print("FINAL EXPERIMENT SUMMARY")
    print("="*50)
    print(f"Baseline Test Loss (No Transforms):  {base_test:.4f}")
    print(f"Augmented Test Loss (With Transforms): {aug_test:.4f}")
    
    plot_comparative_losses(base_train, base_val, aug_train, aug_val)

if __name__ == "__main__":
    main()