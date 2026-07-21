import torch
import tqdm

from utils.dataloader import VOCDataLoader, VOCDataLoaderPerson
from tinyyolov2 import TinyYoloV2
from utils.loss import YoloLoss

loader = VOCDataLoaderPerson(split="train", batch_size=128)
loader_test = VOCDataLoaderPerson(split="val", batch_size=1)

DEVICE = 'cuda'

# make instance with 20 classes as output
net = TinyYoloV2(num_classes=1)

# load pretrained weights
sd = torch.load("state_dicts/voc_pretrained.pt")
net.load_state_dict({k: v for k, v in sd.items() if not '9' in k}, strict=False)

net = net.to(DEVICE)
net.eval()

criterion = YoloLoss(anchors=net.anchors)

# Freeze all layers (we only want to retrain the last one)
for key, param in net.named_parameters():

    if '9' in key:
        param.requires_grad = True
    else:
        param.requires_grad = False


optimizer = torch.optim.Adam(filter(lambda x: x.requires_grad, net.parameters()), lr=0.001) #type: ignore


from utils.ap import precision_recall_levels, ap, display_roc
from utils.yolo import nms, filter_boxes
from utils.viz import display_result

NUM_TEST_SAMPLES = 350
NUM_EPOCHS = 20

test_AP = []


for epoch_n in range(NUM_EPOCHS):

    net.train()
    for idx, (inputs, targets) in tqdm.tqdm(enumerate(loader), total=len(loader)):

        optimizer.zero_grad()

        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

        outputs = net(inputs, yolo=False)
        loss,_ = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

    test_precision = []
    test_recall = []

    net.eval()
    with torch.no_grad():
        for idx, (inputs, targets) in tqdm.tqdm(enumerate(loader_test), total=NUM_TEST_SAMPLES):
            
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            outputs = net(inputs, yolo=True)
            
            #The right threshold values can be adjusted for the target application
            outputs = filter_boxes(outputs, 0.25)
            outputs = nms(outputs, 0.5)
            
            precision, recall = precision_recall_levels(targets[0], outputs[0])
            test_precision.append(precision)
            test_recall.append(recall)
            if idx == NUM_TEST_SAMPLES:
                break
            
            # display_result(inputs.cpu(), outputs, targets.cpu(), file_path=f'yolo_prediction_{idx}.png')
                
    #Calculation of average precision with collected samples
    test_AP.append(ap(test_precision, test_recall))
    print('average precision', test_AP)

    #plot ROC
    #display_roc(test_precision, test_recall)

import matplotlib.pyplot as plt
import numpy as np

epochs = range(0, NUM_EPOCHS)

plt.figure(figsize=(10, 6))

# Plot the main AP progression line
plt.plot(range(0, NUM_EPOCHS), test_AP, marker='o', color='teal', linewidth=2, 
         markersize=6, label='Validation AP')

# Automatically find and highlight the peak performance
best_epoch_idx = np.argmax(test_AP)
best_epoch = epochs[best_epoch_idx]
best_ap = test_AP[best_epoch_idx]

# Plot a distinct marker over the best epoch
plt.plot(best_epoch, best_ap, marker='*', color='gold', markersize=15, 
         markeredgecolor='black', label=f'Best AP ({best_ap:.3f} at Epoch {best_epoch})')

# Formatting and labels
plt.title('Average Precision (AP) Over Training Epochs', fontsize=14, pad=15)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Average Precision (AP)', fontsize=12)

# Lock the Y-axis to standard 0-1 range for metrics
plt.ylim([0.0, 1.05])

# Force the X-axis to only show integer ticks (you can't have half an epoch)
# If you have hundreds of epochs, you might want to comment this out so it doesn't clutter
plt.xticks(epochs)

# Add a subtle grid to help track values across the plot
plt.grid(True, linestyle='--', alpha=0.6)

# Place legend where it typically won't block the curve (lower right for learning curves)
plt.legend(loc='lower right', fontsize=11)

plt.tight_layout()
plt.savefig('results/person_only_detection.png')

torch.save(net.state_dict(), 'state_dicts/yolov2_person_finetuned.pt')

# Export as onnx model
net.eval()

export_input, _ = next(iter(loader_test))
export_input = export_input.to(DEVICE)

torch.onnx.export(
        net,
        export_input,
        "models/person_only_yolo.onnx",
        export_params=True,
        opset_version=11,
        input_names=['input_image'],
        output_names=['yolo_output'],

        # dynamic axes so the model can be used with different batch sizes
        dynamic_axes={'input_image': {0: 'batch_size'}, 'yolo_output': {0: 'batch_size'}}
)