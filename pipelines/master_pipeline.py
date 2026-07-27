import torch

from tinyyolov2 import TinyYoloV2

from pipelines.base_pipeline import BasePipeline

# make pyright shut up
from torch.utils.data import DataLoader
from typing_extensions import Optional

from transformation.transform_generator import TransformGenerator

class MasterPipeline(BasePipeline):
    """
    Master pipeline for applying person only detection training with augmented data.
    """
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
        """
        Initializes TinyYOLOv2 and loads the pretrained VOC weights.
        Checks for dataset validity (person-only) and strips last layer for retraining.
        """

        # Check if the trainings data is person only
        dataset = self.train_loader.dataset
        while hasattr(dataset, 'dataset'):
            dataset = getattr(dataset, 'dataset')
            
        transform_pipeline = getattr(dataset, 'transforms', None) or getattr(dataset, 'transform', None)
        person_only = getattr(transform_pipeline, 'only_person', False)

        if not person_only:
            raise ValueError(
                f"[{self.pipeline_name}] Architecture Mismatch Error: "
                "PersonOnlyPipeline requires a person-only dataloader (e.g., VOCDataLoaderPerson), "
                "but a standard multi-class dataloader was provided!"
            )

        model = TinyYoloV2(num_classes=1)
        
        if self.sd_path is None:
            print(f"[*] No weights loaded. Training fresh network.")
            return model.to(self.device)

        path_to_load = self.sd_path

        print(f"[*] Loading weights from: '{path_to_load}'")
        state_dict = torch.load(path_to_load, map_location=self.device)

        # load pretrained weights but skip 9th layer as we have to retrain it 
        model.load_state_dict({k: v for k, v in state_dict.items() if not '9' in k}, strict=False)

        # Freeze all layers (we only want to retrain the last one)
        for key, param in model.named_parameters():

            if '9' in key:
                param.requires_grad = True
            else:
                param.requires_grad = False

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
