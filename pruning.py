
import torch
torch.rand(1).to('cuda')

import tqdm
import copy
import numpy as np
import time

from utils.dataloader import VOCDataLoader, VOCDataLoaderPerson
from prunedtinyyolov2 import PrunedTinyYoloV2
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


def evaluate_yolo_ap(model: PrunedTinyYoloV2, testloader: DataLoader, device: str, samples: int) -> float:

    test_precision = []
    test_recall = []

    model.eval()

    with torch.no_grad():
        for idx, (inputs, targets) in tqdm.tqdm(enumerate(testloader), total=samples):
            
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs, yolo=True)
            
            #The right threshold values can be adjusted for the target application
            outputs = filter_boxes(outputs, 0.25)
            outputs = nms(outputs, 0.5)
            
            precision, recall = precision_recall_levels(targets[0], outputs[0])
            test_precision.append(precision)
            test_recall.append(recall)
            if idx == samples:
                break
            
    return ap(test_precision, test_recall)
    


def main():

    NUM_TEST_SAMPLES = 350
    NUM_TRAIN_SAMPLES = 350
    NUM_EPOCHS = 20
    NUM_PRUNING_RATIOS = 5

    DEVICE = 'cuda'

    loader = VOCDataLoaderPerson(train=True, batch_size=128)
    loader_test = VOCDataLoaderPerson(train=False, batch_size=1)

    ratios = np.linspace(0, 0.8, NUM_PRUNING_RATIOS)

    # load pretrained weights with person detection only
    state_dict = torch.load('state_dicts/yolov2_person_finetuned.pt')

    ap_untrained = []
    ap_retrained = []
    fps_list = []

    best_ap: int = 0
    best_state_dict: dict = {}

    for idx, ratio in tqdm.tqdm(enumerate(ratios), total=len(ratios), desc="Pruning ratios"):
        
        current_sd = copy.deepcopy(state_dict)
        pruned_sd = l1_structured_pruning(current_sd, ratio)
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

        print(new_channels)

        # make instance with 20 classes as output
        # We use PrunedTinyYolo as channels will change after densifying the state dict
        net = PrunedTinyYoloV2(num_classes=1, channels=new_channels)
        net.load_state_dict(sd)
        net = net.to(DEVICE)
        net.eval()

        # test untrained performance after pruning 
        ap_untrained.append(evaluate_yolo_ap(model=net, testloader=loader_test, device=DEVICE, samples=NUM_TEST_SAMPLES))

        # Prepare retraining loop 
        criterion = YoloLoss(anchors=net.anchors)
        optimizer = torch.optim.Adam(net.parameters(), lr=1e-5)    #type: ignore

        net.train()

        for epoch_n in range(NUM_EPOCHS):
            for idx, (inputs, targets) in tqdm.tqdm(enumerate(loader), total=len(loader), desc="Retraining"):

                optimizer.zero_grad()

                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

                outputs = net(inputs, yolo=False)
                loss,_ = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

        # test retrained performance after epochs finish 
        with torch.no_grad():
            ap_retrained.append(evaluate_yolo_ap(net, loader_test, DEVICE, NUM_TRAIN_SAMPLES))

        # save best performing ratio's state_dict
        if ap_retrained[-1] > best_ap:
            best_ap = ap_retrained[-1]
            best_state_dict = copy.deepcopy(net.state_dict())


        # Measure inference speed
        # We use dummy values since it skips reading from disc, which makes performance metrics more accurate
        # The inference result itself does not matter here
        # 416 x 416 is the dataset image size
        dummy_input = torch.randn(1, 3, 416, 416).to(DEVICE)
        
        with torch.no_grad():
            for _ in range(10):  # Warm-up phase
                _ = net(dummy_input)

        if DEVICE == 'cuda':
            torch.cuda.synchronize()
        start_time = time.time()

        num_runs = 100
        with torch.no_grad():
            for _ in range(num_runs):
                _ = net(dummy_input)

        if DEVICE == 'cuda':
            torch.cuda.synchronize()
        end_time = time.time()

        fps = num_runs / (end_time - start_time)
        fps_list.append(fps)
        print(f"\nRatio {ratio:.2f} -> Base AP: {ap_untrained[-1]:.4f} | FPS: {fps:.2f}")


    # Export best model in onnx format

    new_channels = [
        best_state_dict['conv1.weight'].shape[0],
        best_state_dict['conv2.weight'].shape[0],
        best_state_dict['conv3.weight'].shape[0],
        best_state_dict['conv4.weight'].shape[0],
        best_state_dict['conv5.weight'].shape[0],
        best_state_dict['conv6.weight'].shape[0],
        best_state_dict['conv7.weight'].shape[0],
        best_state_dict['conv8.weight'].shape[0]
    ]

    best_net = PrunedTinyYoloV2(num_classes=1, channels=new_channels)
    best_net.load_state_dict(best_state_dict)
    best_net.to(DEVICE)
    best_net.eval()

    export_input, _ = next(iter(loader_test))
    export_input = export_input.to(DEVICE)

    torch.onnx.export(
        best_net,
        export_input,
        "models/pruned_yolo.onnx",
        export_params=True,
        opset_version=11,
        input_names=['input_image'],
        output_names=['yolo_output'],

        # dynamic axes so the model can be used with different batch sizes
        dynamic_axes={'input_image': {0: 'batch_size'}, 'yolo_output': {0: 'batch_size'}}
    )

    print("Exported the best model to 'models/best_pruned_yolo.onnx'!")
    
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
    plt.savefig('accuracy_recovery.png')
    plt.close()
    print("Graph successfully saved as accuracy_recovery.png!")

    # Inference time over loss plot
    inference_time_ms = [1000.0 / f for f in fps_list]

    plt.figure(figsize=(8, 6))
    plt.plot(ap_retrained, inference_time_ms, marker='o', linestyle='-', color='purple')
    plt.xlabel('Metric (Retrained Average Precision)') 
    plt.ylabel('Inference Time per Image (ms)')
    plt.title('Performance Trade-off: Inference Time vs. Metric')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Annotate points with their pruning ratio
    for i, ratio in enumerate(ratios):
        plt.annotate(f"{ratio:.2f}", 
                     (ap_retrained[i], inference_time_ms[i]), 
                     textcoords="offset points", 
                     xytext=(0,10), 
                     ha='center')

    plt.savefig('inference_vs_metric.png')
    plt.close()


    torch.save(best_net.state_dict(), 'state_dicts/yolov2_pruned.pt')


main()