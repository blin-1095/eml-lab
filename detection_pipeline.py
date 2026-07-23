import time
import cv2
import numpy as np
import torch

from utils.camera import CameraDisplay
from tinyyolov2 import TinyYoloV2
from utils.yolo import filter_boxes, nms
from torchvision import transforms as tf


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

net = TinyYoloV2(num_classes=20)
net.load_state_dict(torch.load("voc_pretrained.pt",
                               map_location=device))
net.to(device)
net.eval()

now = time.time()


import onnxruntime as ort

session = ort.InferenceSession(
    "models/person_only_yolo.onnx",
    providers=[
        "CUDAExecutionProvider",
        "CPUExecutionProvider"
    ]
)

input_name = session.get_inputs()[0].name


def callback(image):

    global now

    fps = f"{int(1/(time.time() - now))}"
    now = time.time()

    image = image[:320, :320, :]

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    """
    x = tf.functional.to_tensor(rgb)
    x = x.unsqueeze(0).to(device)

    #prepro_time = time.time()
    with torch.no_grad():
        output = net(x)
    #"""
    #"""
    x = rgb.transpose(2,0,1).astype(np.float32)
    x /= 255
    input_tensor = np.expand_dims(
        x,
        axis=0
    )
    output = session.run(
        None,
        {
            input_name: input_tensor
        }
    )[0]
    output = torch.from_numpy(output)
    #"""
    #infer_time = time.time()
    output = filter_boxes(output, threshold=0.1)
    output = nms(output, threshold=0.25)

    #post_time = time.time()
    height_width = 320
    for det in output[0]:

        xc, yc, bw, bh, score, cls = det.cpu().numpy()

        x1 = int((xc - bw/2) * height_width)
        y1 = int((yc - bh/2) * height_width)

        x2 = int((xc + bw/2) * height_width)
        y2 = int((yc + bh/2) * height_width)

        cv2.rectangle(image,
                      (x1,y1),
                      (x2,y2),
                      (0,0,255),
                      2)

        cv2.putText(image,
                    f"{score:.2f}",
                    (x1, max(20,y1-5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0,0,255),
                    2)

    cv2.putText(image, "fps="+fps, (2, 25), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (100, 255, 0), 2, cv2.LINE_AA)
    #box_time = time.time()
    #print(prepro_time - now)
    #print(infer_time - now)
    #print(post_time - now)
    #print(box_time - now)
    return image

cam = CameraDisplay(callback)

cam.start()

input = input("Enter 'q' to quit")
if input == "q":
    # Stop when finished
    cam.stop()
    cam.release()