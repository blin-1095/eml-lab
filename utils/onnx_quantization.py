import os
import onnxruntime as ort
from onnxruntime.quantization import quantize_static, QuantType, CalibrationDataReader

class ONNXCalibrationReader(CalibrationDataReader):
    """
    Wraps a PyTorch DataLoader into an ONNX-compatible Calibration Reader.
    Converts batches into dictionaries of NumPy arrays for Static Quantization.
    """
    def __init__(self, dataloader, onnx_model_path: str):

        if ort is None:
            raise ImportError("ONNX Runtime is missing. Run: pip install onnxruntime")

        self.iterator = iter(dataloader)
        
        # Extract the exact input node name from the ONNX graph
        session = ort.InferenceSession(onnx_model_path, providers=['CUDAExecutionProvider'])
        self.input_name = session.get_inputs()[0].name

    def get_next(self) -> dict:
        try:
            images, _ = next(self.iterator)
            return {self.input_name: images.numpy()}
        except StopIteration:
            # Signals to the quantizer that calibration is done
            return None #type: ignore


def apply_static_quantization(input_onnx_path: str, output_onnx_path: str, calib_loader) -> str:
    """
    Applies INT8 Static Quantization using a calibration dataset.
    """
    os.makedirs(os.path.dirname(output_onnx_path), exist_ok=True)
    
    # Initialize the data reader imported from dataloader.py
    data_reader = ONNXCalibrationReader(calib_loader, input_onnx_path)
    
    # Perform Static Quantization
    quantize_static(
        model_input=input_onnx_path,
        model_output=output_onnx_path,
        calibration_data_reader=data_reader,
        quant_format=ort.quantization.QuantFormat.QDQ,  #type: ignore
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QUInt8
    )
    
    orig_size = os.path.getsize(input_onnx_path) / (1024 * 1024)
    quant_size = os.path.getsize(output_onnx_path) / (1024 * 1024)
    reduction = (1 - (quant_size / orig_size)) * 100
    
    print(f"   Static Quantization complete! Saved to: {output_onnx_path}")
    print(f"   Original Size:  {orig_size:.2f} MB")
    print(f"   Quantized Size: {quant_size:.2f} MB")
    print(f"   Reduction:      {reduction:.1f}%")
    
    return output_onnx_path