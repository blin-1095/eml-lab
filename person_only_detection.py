import torch
import tqdm

from utils.dataloader import VOCDataLoader, VOCDataLoaderPerson
from tinyyolov2 import TinyYoloV2
from utils.loss import YoloLoss

loader = VOCDataLoaderPerson(train=True, batch_size=128)
loader_test = VOCDataLoaderPerson(train=False, batch_size=1)

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
NUM_EPOCHS = 5
test_AP = []

net.train()

for epoch_n in range(NUM_EPOCHS):
    for idx, (inputs, targets) in tqdm.tqdm(enumerate(loader), total=len(loader)):

        optimizer.zero_grad()

        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

        outputs = net(inputs, yolo=False)
        loss,_ = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

    test_precision = []
    test_recall = []

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
    display_roc(test_precision, test_recall)

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