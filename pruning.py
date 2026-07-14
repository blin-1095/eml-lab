
import torch
torch.rand(1).to('cuda')

import tqdm
import copy
from typing import Dict, List
import numpy as np

from utils.dataloader import VOCDataLoader, VOCDataLoaderPerson
from tinyyolov2 import TinyYoloV2
from utils.loss import YoloLoss

from utils.ap import precision_recall_levels, ap, display_roc
from utils.yolo import nms, filter_boxes
from utils.viz import display_result


def l1_structured_pruning(state_dict: Dict, prune_ratio: float) -> Dict:

    state_dict = copy.deepcopy(state_dict)

    for i in range(1,10):

        channel_norms = torch.sum(torch.abs(state_dict[f"conv{i}.weight"]), dim=(1,2,3))
        threshold = torch.quantile(channel_norms, prune_ratio)
        mask = (channel_norms >= threshold).float()
        state_dict[f"conv{i}.weight"] = state_dict[f"conv{i}.weight"] * mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        #state_dict[f"conv{i}.bias"] = state_dict[f"conv{i}.bias"] * mask

    return state_dict


def densify_state_dict(state_dict: Dict) -> Dict:

    state_dict = copy.deepcopy(state_dict)

    for i in range(1,10):

        filter_norm = torch.sum(torch.abs(state_dict[f"conv{i}.weight"]), dim=(1,2,3))

        mask = (filter_norm > 0)
        state_dict[f"conv{i}.weight"] = state_dict[f"conv{i}.weight"][mask, :, :, :]
        #state_dict[f"conv{i}.bias"] = state_dict[f"conv{i}.bias"][mask]

        if i != 9:
            state_dict[f"conv{i + 1}.weight"] = state_dict[f"conv{i + 1}.weight"][:, mask, :, :]

    return state_dict


def evaluate_yolo_ap(model: TinyYoloV2, testloader: VOCDataLoader, device: str, samples: int) -> Dict:

    test_precision = []
    test_recall = []

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
    NUM_EPOCHS = 5

    loader = VOCDataLoaderPerson(train=True, batch_size=1)
    loader_test = VOCDataLoaderPerson(train=False, batch_size=1)

    DEVICE = 'cuda'
    ratios = np.linspace(0, 0.8, 15)

    # make instance with 20 classes as output
    net = TinyYoloV2(num_classes=1)

    # load pretrained weights with person detection only
    state_dict = torch.load('yolov2_person_finetuned.pt')
    # net.load_state_dict(state_dict)
    net = net.to(DEVICE)
    net.eval()

    ap_dense, idxs = [], []
    for idx, ratio in tqdm.tqdm(enumerate(ratios), total=len(ratios)):

        current_sd = copy.deepcopy(state_dict)
        pruned_sd = l1_structured_pruning(current_sd, ratio)
        sd = densify_state_dict(pruned_sd)

        net.load_state_dict(sd)

        ap_dense.append(evaluate_yolo_ap(model=net, testloader=loader_test, device=DEVICE, samples=NUM_TEST_SAMPLES))
        idxs.append(idx)
    
    # Plot the SP values before retraining and with different pruning levels
    #plot([(idxs, ap_dense, 'ap_dense')], xlabel='idxs', save_path='accuracy_l1.png')

    import matplotlib.pyplot as plt
    # Replace the failing plot() line with this:
    plt.plot(ratios, ap_dense, label='AP Dense', marker='o')
    plt.xlabel('Pruning Ratios')
    plt.ylabel('Average Precision')
    plt.legend()
    plt.savefig('accuracy_l1.png')
    print("Graph successfully saved as accuracy_l1.png!")
    

    criterion = YoloLoss(anchors=net.anchors)

    # Freeze all layers (we only want to retrain the last one)
    for key, param in net.named_parameters():

        if '9' in key:
            param.requires_grad = True
        else:
            param.requires_grad = False


    optimizer = torch.optim.Adam(filter(lambda x: x.requires_grad, net.parameters()), lr=0.001)

    net.train()

    test_ap = []

    for epoch_n in range(NUM_EPOCHS):
        for idx, (inputs, targets) in tqdm.tqdm(enumerate(loader), total=len(loader)):

            optimizer.zero_grad()

            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

            outputs = net(inputs, yolo=False)
            loss,_ = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            test_ap.append(evaluate_yolo_ap(net, loader_test, DEVICE, NUM_TEST_SAMPLES))

        #plot ROC
        display_roc(test_precision, test_recall)

    torch.save(net.state_dict(), 'yolov2_person_finetuned.pt')




main()