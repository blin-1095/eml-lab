import torch

from tinyyolov2 import TinyYoloV2

from pipelines.base_pipeline import BasePipeline

# make pyright shut up
from torch.utils.data import DataLoader
from typing_extensions import Optional

class BaselinePipeline(BasePipeline):
    
    def __init__(
        self, 
        pipeline_name: str, 
        train_loader: DataLoader, 
        val_loader: DataLoader, 
        test_loader: DataLoader, 
        device: torch.device, 
        learning_rate: float, 
        epochs: int, 
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
            patience=patience
        )

    def _setup_model(self) -> torch.nn.Module:
        """Initializes TinyYOLOv2 and loads the pretrained VOC weights."""
        model = TinyYoloV2(num_classes=20)
        
        if self._state_dict is None:
            print(f"[*] No weights loaded. Training fresh network.'")
            return model.to(self.device)

        print(f"[*] Loading weights")
        state_dict = self._state_dict
        
        model.load_state_dict(state_dict)
        return model.to(self.device)

    def _preprocess_batch(self, images: torch.Tensor, targets: torch.Tensor):
        """No preprocessing required in the baseline model"""
            
        return images, targets
