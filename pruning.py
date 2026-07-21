
import torch

torch.rand(1).to('cuda')

import tqdm
import copy
import numpy as np
import time

from utils.dataloader import VOCDataLoader, VOCDataLoaderPerson
from tinyyolov2 import TinyYoloV2
from utils.loss import YoloLoss

from utils.ap import precision_recall_levels, ap, display_roc
from utils.yolo import nms, filter_boxes
from utils.viz import display_result

# Stuff to make pyright shut up
from torch.utils.data import DataLoader
from typing import Dict, List


def l1_structured_pruning(state_dict: Dict, prune_ratio: float) -> Dict:

    state_dict = copy.deepcopy(state_dict)

    # only 1-8 because we dont want to pruned the ninth layer
    for i in range(1,9):

        channel_norms = torch.sum(torch.abs(state_dict[f"conv{i}.weight"]), dim=(1,2,3))
        threshold = torch.quantile(channel_norms, prune_ratio)
        mask = (channel_norms >= threshold).float()
        state_dict[f"conv{i}.weight"] = state_dict[f"conv{i}.weight"] * mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        # prune batchnorm layers as well
        state_dict[f"bn{i}.weight"] = state_dict[f"bn{i}.weight"] * mask
        state_dict[f"bn{i}.bias"] = state_dict[f"bn{i}.bias"] * mask
        state_dict[f"bn{i}.running_mean"] = state_dict[f"bn{i}.running_mean"] * mask
        state_dict[f"bn{i}.running_var"] = state_dict[f"bn{i}.running_var"] * mask

    return state_dict


def densify_state_dict(state_dict: Dict) -> Dict:

    state_dict = copy.deepcopy(state_dict)

    for i in range(1,10):

        filter_norm = torch.sum(torch.abs(state_dict[f"conv{i}.weight"]), dim=(1,2,3))

        mask = (filter_norm > 0)
        state_dict[f"conv{i}.weight"] = state_dict[f"conv{i}.weight"][mask, :, :, :]

        if i != 9:
            state_dict[f"conv{i + 1}.weight"] = state_dict[f"conv{i + 1}.weight"][:, mask, :, :]

        bn_key = f"bn{i}"
        if f"{bn_key}.weight" in state_dict:
            for param in ['weight', 'bias', 'running_mean', 'running_var']:
                full_key = f"{bn_key}.{param}"
                state_dict[full_key] = state_dict[full_key][mask]

    return state_dict


def calculate_ap(model: TinyYoloV2, testloader: DataLoader, device: str, samples: int) -> float:

    test_precision = []
    test_recall = []

    model.eval()

    with torch.no_grad():
        for idx, (inputs, targets) in tqdm.tqdm(enumerate(testloader), total=samples, desc="Evaluating AP"):
            
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs, yolo=True)
            
            #The right threshold values can be adjusted for the target application
            outputs = filter_boxes(outputs, 0.25)

            # Select the top 100 boxes if we have too many boxes
            # This is needed to avoid inference degradation when the network is heavily pruned, 
            # as it will start detecting random noise as objects and spawn too many boxes
            if len(outputs[0]) > 50:
                # Sort descending by confidence score
                outputs[0] = outputs[0][outputs[0][:, 4].argsort(descending=True)]
                # Slice the top 100
                outputs[0] = outputs[0][:50]

            outputs = nms(outputs, 0.5)
            
            precision, recall = precision_recall_levels(targets[0], outputs[0])
            test_precision.append(precision)
            test_recall.append(recall)
            if idx == samples:
                break
            
    return ap(test_precision, test_recall)

def calculate_loss(model: TinyYoloV2, criterion: YoloLoss, testloader: DataLoader, device: str, samples: int) -> float:

    model.eval()
    total_val_loss = 0.0
    
    with torch.no_grad():
        # 1. Calculate Validation Loss (Requires raw outputs)
        for idx, (inputs, targets) in tqdm.tqdm(enumerate(testloader), total=samples, desc="Evaluating Loss"):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs_raw = model(inputs, yolo=False)
            loss, _ = criterion(outputs_raw, targets)
            total_val_loss += loss.item()
            
        return total_val_loss / len(testloader)


def measure_inference_speed(model: TinyYoloV2, device: str, num_runs: int = 2000):
    # Measure inference speed
    # We use dummy values since it skips reading from disc, which makes performance metrics more accurate
    # The inference result itself does not matter here
    # 416 x 416 is the dataset image size
    dummy_input = torch.randn(1, 3, 416, 416).to(device)

    with torch.no_grad():
        for _ in range(10):  # Warm-up phase
            _ = model(dummy_input)

    if device == 'cuda':
        torch.cuda.synchronize()
    start_time = time.time()

    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(dummy_input)

    if device == 'cuda':
        torch.cuda.synchronize()
    end_time = time.time()

    fps = num_runs / (end_time - start_time)

    tqdm.tqdm.write(f"\nFPS: {fps:.2f}")
    
    return fps 

def export_model(state_dict: dict, loader: DataLoader, dest: str, device: str):

    sd = copy.deepcopy(state_dict)

    # Export best model in onnx format
    new_channels = [
        sd['conv1.weight'].shape[0],
        sd['conv2.weight'].shape[0],
        sd['conv3.weight'].shape[0],
        sd['conv4.weight'].shape[0],
        sd['conv5.weight'].shape[0],
        sd['conv6.weight'].shape[0],
        sd['conv7.weight'].shape[0],
        sd['conv8.weight'].shape[0]
    ]

    best_net = TinyYoloV2(num_classes=1, channels=new_channels)
    best_net.load_state_dict(state_dict)
    best_net.to(device)
    best_net.eval()

    export_input, _ = next(iter(loader))
    export_input = export_input.to(device)

    torch.onnx.export(
        best_net,
        export_input,
        dest,
        export_params=True,
        opset_version=11,
        input_names=['input_image'],
        output_names=['yolo_output'],

        # dynamic axes so the model can be used with different batch sizes
        dynamic_axes={'input_image': {0: 'batch_size'}, 'yolo_output': {0: 'batch_size'}}
    )

    print("Exported the best model to 'models/best_pruned_yolo.onnx'!")


def main():

    NUM_EPOCHS = 20
    NUM_PRUNING_RATIOS = 5

    DEVICE = 'cuda'

    loader_train = VOCDataLoaderPerson(split="train", batch_size=196)
    loader_test = VOCDataLoaderPerson(split="test", batch_size=1)
    loader_val = VOCDataLoaderPerson(split="val", batch_size=1)

    ratios = np.linspace(0, 0.8, NUM_PRUNING_RATIOS)

    # load pretrained weights with person detection only
    state_dict = torch.load('state_dicts/yolov2_person_finetuned.pt')

    ap_untrained = []
    ap_retrained = []
    loss_untrained = []
    loss_retrained = []
    fps_list = []

    best_score: float = float('inf')
    baseline_fps: float = 1.0  # Placeholder to prevent division errors
    best_state_dict: dict = {}

    current_sd = copy.deepcopy(state_dict)
    current_pruned_ratio: float = 0.0

    for idx, target_ratio in tqdm.tqdm(enumerate(ratios), total=len(ratios), desc="Pruning ratios"):

        # variables to save best weights for current ratio
        lowest_ratio_loss: float = float('inf')
        best_ratio_state_dict: dict = {}

        # calculate current pruning ratio
        if target_ratio == 0.0:
            relative_ratio = 0.0
        else:
            relative_ratio = (target_ratio - current_pruned_ratio) / (1.0 - current_pruned_ratio)
            
        tqdm.tqdm.write(f"\nTarget Sparsity: {target_ratio:.2f} | Pruning {relative_ratio*100:.1f}% of remaining channels")

        pruned_sd = l1_structured_pruning(current_sd, relative_ratio)
        sd = densify_state_dict(pruned_sd)

        new_channels = [
            sd['conv1.weight'].shape[0],
            sd['conv2.weight'].shape[0],
            sd['conv3.weight'].shape[0],
            sd['conv4.weight'].shape[0],
            sd['conv5.weight'].shape[0],
            sd['conv6.weight'].shape[0],
            sd['conv7.weight'].shape[0],
            sd['conv8.weight'].shape[0]
        ]

        # make instance with 20 classes as output
        # We use PrunedTinyYolo as channels will change after densifying the state dict
        net = TinyYoloV2(num_classes=1, channels=new_channels)
        net.load_state_dict(sd)
        net = net.to(DEVICE)

        # Prepare retraining loop 
        criterion_train = YoloLoss(anchors=net.anchors)

        criterion_eval = YoloLoss(anchors=net.anchors)
        criterion_eval.seen = 999999

        optimizer = torch.optim.Adam(net.parameters(), lr=1e-5)    #type: ignore

        # test untrained performance after pruning 
        ap_untrained.append(calculate_ap(model=net, testloader=loader_test, device=DEVICE, samples=len(loader_test)))
        loss_untrained.append(calculate_loss(net, criterion_eval, loader_test, DEVICE, len(loader_test)))

        if target_ratio == 0.0:
            tqdm.tqdm.write("Ratio 0.0 detected. Skipping retraining to preserve baseline weights.")
            # The retrained AP is exactly the same as the untrained AP
            ap_retrained.append(ap_untrained[-1])
            loss_retrained.append(loss_untrained[-1])
            best_ratio_state_dict = copy.deepcopy(net.state_dict())

        else:
            PATIENCE = 5
            epochs_without_improve = 0

            for epoch_n in range(NUM_EPOCHS):

                # TRAINING
                net.train()
                for idx, (inputs, targets) in tqdm.tqdm(enumerate(loader_train), total=len(loader_train), desc="Retraining"):

                    optimizer.zero_grad()

                    inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

                    outputs = net(inputs, yolo=False)
                    loss,_ = criterion_train(outputs, targets)
                    loss.backward()
                    optimizer.step()


                # VALIDATION
                net.eval()
                with torch.no_grad():
                    current_loss = calculate_loss(net, criterion_eval, loader_val, DEVICE, len(loader_val))

                tqdm.tqdm.write(f"Epoch {epoch_n} | Validation Loss: {current_loss:.4f}")

                if current_loss < lowest_ratio_loss:
                    lowest_ratio_loss = current_loss
                    best_ratio_state_dict = copy.deepcopy(net.state_dict())
                    epochs_without_improve = 0

                else:
                    # No improvement this epoch
                    epochs_without_improve += 1

                if epochs_without_improve >= PATIENCE:
                    tqdm.tqdm.write(f"Early stopping triggered! Validation Loss hasn't improved for {PATIENCE} epochs.")
                    tqdm.tqdm.write(f"Breaking out of training at epoch {epoch_n}. Lowest Val Loss was: {lowest_ratio_loss:.4f}")
                    break


            # TESTING
            # test retrained performance on test set after epochs finish
            net.load_state_dict(best_ratio_state_dict)

            with torch.no_grad():
                ap_retrained.append(calculate_ap(net, loader_test, DEVICE, len(loader_test)))
                tqdm.tqdm.write(f"Final Test AP for Ratio {target_ratio:.2f}: {ap_retrained[-1]:.4f}")

                loss_retrained.append(calculate_loss(net, criterion_eval, loader_test, DEVICE, len(loader_test)))
                tqdm.tqdm.write(f"Final Test Loss for Ratio {target_ratio:.2f}: {loss_retrained[-1]:.4f}")


            # update state dict for next pruning ratio loop
            current_sd = copy.deepcopy(best_ratio_state_dict)
            current_pruned_ratio = target_ratio

        fps_list.append(measure_inference_speed(net, DEVICE))

        test_loss = loss_retrained[-1]

        # calculate loss efficiency score
        if target_ratio == 0.0:
            baseline_fps = fps_list[-1]
            current_score = test_loss
        else:
            speedup_factor = fps_list[-1] / baseline_fps
            # Lower score is better!
            current_score = test_loss / speedup_factor

        tqdm.tqdm.write(f"Loss Efficiency Score: {current_score:.4f}")

        # select best model based on efficiency score
        if current_score < best_score:
            print(f"*** New best model! Score increased to {current_score:.4f} from {best_score:.4f} ***")
            best_score = current_score
            best_state_dict = copy.deepcopy(best_ratio_state_dict)


    export_model(best_state_dict, loader_test, "models/pruned_yolo.onnx", DEVICE)
    
    # Plot the graphs
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(8, 6))
    
    # Accuracy recovery plot
    plt.plot(ratios, ap_untrained, label='Untrained Pruned AP', marker='o', color='blue')
    plt.plot(ratios, ap_retrained, label='Retrained (Recovered) AP', marker='s', color='green')
    
    plt.xlabel('Pruning Ratios')
    plt.ylabel('Average Precision (AP)')
    plt.title('YOLOv2 Pruning: Before and After Retraining')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig('results/accuracy_recovery.png')
    plt.close()
    print("Graph successfully saved as accuracy_recovery.png!")


    inference_time_ms = [1000.0 / f for f in fps_list]

    # Pruning Ratio vs. Inference Time
    plt.figure(figsize=(8, 6))
    plt.plot(ratios, inference_time_ms, marker='o', linestyle='-', color='blue')
    plt.xlabel('Pruning Ratio') 
    plt.ylabel('Inference Time per Image (ms)')
    plt.title('Pruning Ratio vs. Inference Time')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.savefig('results/ratio_vs_inference_time.png')
    plt.close()

    # Pruning Ratio vs. Average Precision
    plt.figure(figsize=(8, 6))
    plt.plot(ratios, ap_retrained, marker='o', linestyle='-', color='green')
    plt.xlabel('Pruning Ratio') 
    plt.ylabel('Average Precision (Retrained)')
    plt.title('Pruning Ratio vs. Average Precision')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.savefig('results/ratio_vs_ap.png')
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.plot(ratios, loss_retrained, marker='o', linestyle='-', color='red')
    plt.xlabel('Pruning Ratio') 
    plt.ylabel('Test Loss')
    plt.title('Pruning Ratio vs Test Loss')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig('results/loss_train.png')
    plt.close()


    torch.save(best_state_dict, 'state_dicts/yolov2_pruned.pt')


main()