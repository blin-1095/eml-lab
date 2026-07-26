from utils.camera import CameraDisplay
import time

class BaseDetectionPipeline():

    def __init__(self):
        self.frame_count = 0
        self.warmup_frames = 30
        self.start_time = None
        self.frame_threshold = 1000 + self.warmup_frames

    def callback(self, image):
        self.frame_count += 1
        return image

    def run_pipeline(self):
        """
        Runs only the model pipeline. Enter 'q' to stop it.
        """
        cam = CameraDisplay(self.callback)

        cam.start()

        while input("Enter 'q' to quit: ") != "q":
            pass

        cam.stop()
        cam.release()

    def fps_pipeline(self):
        """
        Runs the model pipeline and measures the fps. Stops after reaching self.frame_threshold.
        """
        self.frame_count = 0
        self.start_time = time.perf_counter()

        cam = CameraDisplay(self.callback)
        cam.start()

        while self.frame_count < self.frame_threshold:
            pass

        cam.stop()
        cam.release()

        elapsed = time.perf_counter() - self.start_time
        measured_frames = self.frame_count - self.warmup_frames
        avg_fps = measured_frames / elapsed

        return avg_fps

