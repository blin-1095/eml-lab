import torch
import time
import cv2
import numpy as np
import onnxruntime as ort

from base_detection_pipeline import BaseDetectionPipeline
from utils.yolo import nms, filter_boxes

class OnnxDetectionPipeline(BaseDetectionPipeline):

    def __init__(self, onnx_path, device="cuda"):
        super().__init__()
        self.session = ort.InferenceSession(
            onnx_path,
            providers=["CUDAExecutionProvider"]
        )

        input_name = self.session.get_inputs()[0].name
        output_name = self.session.get_outputs()[0].name

        self.input_tensor = torch.empty(
            (1, 3, 320, 320),
            device=device,
            dtype=torch.float32
        )

        self.output_tensor = torch.empty(
            (1, 5, 10, 10, 6),
            device=device,
            dtype=torch.float32
        )

        self.io_binding = self.session.io_binding()

        self.io_binding.bind_input(
            name=input_name,
            device_type=device,
            device_id=0,
            element_type=np.float32,
            shape=tuple(self.input_tensor.shape),
            buffer_ptr=self.input_tensor.data_ptr(),
        )

        self.io_binding.bind_output(
            name=output_name,
            device_type=device,
            device_id=0,
            element_type=np.float32,
            shape=tuple(self.output_tensor.shape),
            buffer_ptr=self.output_tensor.data_ptr(),
        )

        self.input_image = np.empty((1,3,320,320), dtype=np.float32)
        self.prev_time = time.time()

    def callback(self, image):
        if self.frame_count == self.warmup_frames:
            self.start_time = time.time()
        self.frame_count += 1
        fps = f"{int(1/(time.time() - self.prev_time))}"
        self.prev_time = time.time()

        image = image[:320, :320, :]

        self.input_image[0] = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).transpose(2,0,1)
        self.input_image *= 1/255.0

        self.input_tensor.copy_(torch.from_numpy(self.input_image))

        #prepro_time = time.time()

        self.session.run_with_iobinding(self.io_binding)
        output = self.output_tensor

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