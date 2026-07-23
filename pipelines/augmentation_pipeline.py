import torch

from tinyyolov2 import TinyYoloV2

from pipelines.base_pipeline import BasePipeline

# make pyright shut up
from torch.utils.data import DataLoader
from typing_extensions import Optional
from transformation.transform_generator import TransformGenerator

class AugmentationPipeline(BasePipeline):
    def __init__(
        self, 
        pipeline_name: str, 
        train_loader: DataLoader, 
        val_loader: DataLoader, 
        test_loader: DataLoader, 
        device: torch.device, 
        learning_rate: float, 
        epochs: int, 
        transform_generator: TransformGenerator,
        sd_path: Optional[str] = None, 
        patience: int = 20
    ):
        super().__init__(
            pipeline_name=pipeline_name,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            device=device,
            learning_rate=learning_rate,
            epochs=epochs,
            sd_path=sd_path,
            patience=patience
        )

        self.transform_generator = transform_generator
        self.total_params = sum([t.get_required_params_amount() for t in self.transform_generator.get_transformations()])

    def _setup_model(self) -> torch.nn.Module:
        """Initializes TinyYOLOv2 and loads the pretrained VOC weights."""
        model = TinyYoloV2(num_classes=20)
        
        if self.sd_path is None:
            print(f"[*] No weights loaded. Training fresh network.'")
            return model.to(self.device)

        path_to_load = self.sd_path
        
        print(f"[*] Loading weights from: '{path_to_load}'")
        state_dict = torch.load(path_to_load, map_location=self.device)
        
        model.load_state_dict(state_dict)
        return model.to(self.device)

    def _preprocess_batch(self, images: torch.Tensor, targets: torch.Tensor):
        """Actively manipulates the batch with geometric/photometric augmentations."""
        if self.transform_generator is not None:
            batch_size = images.shape[0]
            
            # Generate the random dice rolls for the parameter values
            random_params = torch.rand((batch_size, self.total_params), device=self.device)
            
            # Apply the transformations
            images, targets = self.transform_generator.transform(random_params, images=images, targets=targets)

        return images, targets
