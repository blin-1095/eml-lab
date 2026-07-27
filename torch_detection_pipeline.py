import torch
import time
import cv2
from torchvision import transforms as tf
from tinyyolov2 import TinyYoloV2

from base_detection_pipeline import BaseDetectionPipeline
from utils.yolo import nms, filter_boxes

class TorchDetectionPipeline(BaseDetectionPipeline):

    def __init__(self, device):
        super().__init__(device)
        self.net = None
        #self.net = TinyYoloV2(num_classes=num_classes)
        #self.net.load_state_dict(torch.load(state_dict_path, map_location=self.device))
        self.prev_time = time.time()

    def setup_model(self, num_classes, state_dict_path):
        self.net = TinyYoloV2(num_classes=num_classes)
        self.net.load_state_dict(torch.load(state_dict_path, map_location=self.device))
        self.prev_time = time.time()

    def callback(self, image):
        if self.frame_count == self.warmup_frames:
            self.start_time = time.time()

        self.frame_count += 1
        fps = f"{int(1/(time.time() - self.prev_time))}"
        self.prev_time = time.time()
    
        image = image[:320, :320, :]
    
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        x = tf.functional.to_tensor(rgb)
        x = x.unsqueeze(0).to(self.device)
    
        #prepro_time = time.time()

        with torch.no_grad():
            output = self.net(x)

        #infer_time = time.time()

        output = filter_boxes(output, threshold=0.1)
        output = nms(output, threshold=0.25)
    
        #post_time = time.time()

        image_size = 320
        detections = output[0].cpu().numpy()
        for det in detections:
    
            xc, yc, bw, bh, score, cls = det
    
            x1 = int((xc - bw/2) * image_size)
            y1 = int((yc - bh/2) * image_size)
    
            x2 = int((xc + bw/2) * image_size)
            y2 = int((yc + bh/2) * image_size)
    
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
        #print(prepro_time - self.prev_time)
        #print(infer_time - self.prev_time)
        #print(post_time - self.prev_time)
        #print(box_time - self.prev_time)
        return image