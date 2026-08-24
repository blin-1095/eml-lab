# Setup

## uv
This project is managed using **uv**.
You need to first install **uv** to be able to initialize the python environment.

After installing uv execute

```
uv sync
```

After that you can source the venv as usual and execute the scripts.

## TensorRT

Don't forget to install TensorRT!

# General notes

## Best pruned model

We currently use the 20% pruned model for further optimizations.
A general measure for best pruning ratio shall be introduced sometime in the future.

## ONNX Quantization

We use static quantization for the following reasons:

TensorRT Engine Compatibility: The TensorrtExecutionProvider on Jetson cannot natively fuse or optimize dynamically quantized models. Static quantization (using QDQ nodes) creates a graph structure that TensorRT can fully parse, compile, and optimize into a high-performance execution engine.

Elimination of Runtime Overhead: Dynamic quantization computes activation scales on the fly during inference, introducing extra computational steps. Static quantization bakes these scaling factors directly into the graph, saving valuable GPU cycles on the edge.

True Activation Compression: Dynamic quantization only compresses model weights, leaving activations as FP32 until runtime where they are converted on the fly. Static quantization compresses both weights and activations ahead of time. This significantly reduces memory bandwidth requirements.

Hardware Tensor Core Acceleration: Jetson's integrated GPU features specialized Tensor Cores designed for fast INT8 matrix multiplication. These cores require fixed, static quantization parameters to operate at peak efficiency.

Jetson DLA (Deep Learning Accelerator) Support: If your Jetson model targets the hardware Deep Learning Accelerator (DLA) cores to save power, static quantization is strictly mandatory, as the DLA does not support dynamic operations.

### Using TensorRT provider

Using standard CUDA provider leads to this:

'''
2026-07-27 12:53:13.725899659 [W:onnxruntime:, transformer_memcpy.cc:74 ApplyImpl] 11 Memcpy nodes are added to the graph main_graph for CUDAExecutionProvider. It might have negative impact on performance (including unable to run CUDA graph). Set session_options.log_severity_level=1 to see the detail logs before this message.
2026-07-27 12:53:13.726878409 [W:onnxruntime:, session_state.cc:1166 VerifyEachNodeIsAssignedToAnEp] Some nodes were not assigned to the preferred execution providers which may or may not have an negative impact on performance. e.g. ORT explicitly assigns shape related ops to CPU to improve perf.
2026-07-27 12:53:13.726891083 [W:onnxruntime:, session_state.cc:1168 VerifyEachNodeIsAssignedToAnEp] Rerunning with verbose output on a non-minimal build will show node assignments.
'''

We therefore use TensorrtExecutionProvider instead.
TensorRT is optimized for INT8 inference. CUDA is designed for FLOAT32 training and inference.

### Summary of Format Selection
We utilize the **QDQ (Quantize-Dequantize)** format because it acts as an *Intermediate Representation* rather than a rigid execution graph. By preserving the original floating-point topology of our YOLO architecture and supplying INT8 scaling factors as separate nodes, we give hardware-specific compilers maximum flexibility. 

When this QDQ graph is executed on an Nvidia Jetson, ONNX Runtime (and specifically the TensorRT Execution Provider) reads the scaling hints, strips out the Q/DQ nodes, performs advanced layer fusion, and dynamically compiles native, ultra-fast INT8 execution kernels specifically tailored to the Jetson's silicon.

## Transforms

In this implementation, each transformation of an image has a set chance to be applied on the image. Because TinyYoloV2 is a very small network, applying all transformations at once will lead to underfitting, as the network will struggle to make sense of the heavily transformed data.

### Augmentation Probability Breakdown

To strike the perfect balance between robust regularization and model capacity, each geometric and photometric transformation in this pipeline is applied independently with a **15% probability**. 

When multiple augmentations are active (e.g., 9 distinct transforms), a high individual probability can stack unpredictably, generating heavily distorted, "unsolvable" images that stall training. By lowering the independent trigger chance to 15%, we successfully control the compound probability. 

Given **9 active transformations**, the statistical distribution of augmentations per image across a training batch breaks down as follows (based on a binomial distribution):

| Transforms Applied | Probability | Impact on Training Batch |
| :---: | :---: | :--- |
| **0** | **23.16%** | Image remains perfectly clean. Anchors the model to baseline reality. |
| **1** | **36.79%** | Single distortion. Forces the model to learn specific invariances (e.g., just noise or just rotation). |
| **2** | **25.97%** | Moderate stacking. Introduces harder edge cases without completely destroying object semantics. |
| **3** | **10.69%** | Heavy stacking. Acts as a severe regularization test to strictly prevent overfitting. |
| **4** | **2.83%** | Extreme distortion. Very chaotic images that force the model to look at partial features. |
| **5** | **0.50%** | Near-total loss of semantics. Rare enough that it doesn't stall the optimizer. |
| **6 to 9** | **< 0.1%** | Total chaos. Statistically insignificant occurrence. |

**The Result:** Nearly **60%** of the images in any given batch will receive exactly 0 or 1 transformation, providing a healthy, solvable baseline for the optimizer. Another **26%** will receive exactly 2 transforms to push the model's geometric boundaries, while extreme edge cases (3 or more) are kept strictly below 15% to prevent the dataset from becoming unsolvable noise.



### Rotations

During data augmentation, applying random rotations to training images causes severe bounding box inflation. Because object detection models require axis-aligned bounding boxes (AABBs), calculating the minimum and maximum coordinates of rotated box corners inherently expands their footprint, causing boxes to bloat and distort across the frame. Rotations are therefore limited to $\pm10^\circ$.