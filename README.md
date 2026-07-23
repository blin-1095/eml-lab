# Setup

This project is managed using **uv**.
You need to first install **uv** to be able to initialize the python environment.

After installing uv execute

```
uv sync
```

After that you can source the venv as usual and execute the scripts.



# General notes

## Efficiency score

To select the best performing model after pruning, we have to take into account the Loss and the achieved performance gain.
Loss alone is not enough as the metric, since the lowest Loss model will always be the unpruned one (or models that have been pruned less).

To get the best performing model per FPS, we use the following equation:

$$
\text{Efficiency Score} = \frac{\text{Validation Loss}}{\left( \frac{\text{FPS}_{\text{current}}}{\text{FPS}_{\text{baseline}}} \right)}
$$

This calculates an Efficiency Score which allows us to compare whether the rise in Loss is worth the speedup gain from pruning.
A lower score is better.

## ONNX Quantization

We use static quantization for the following reasons:

TensorRT Engine Compatibility: The TensorrtExecutionProvider on Jetson cannot natively fuse or optimize dynamically quantized models. Static quantization (using QDQ nodes) creates a graph structure that TensorRT can fully parse, compile, and optimize into a high-performance execution engine.

Elimination of Runtime Overhead: Dynamic quantization computes activation scales on the fly during inference, introducing extra computational steps. Static quantization bakes these scaling factors directly into the graph, saving valuable GPU cycles on the edge.

True Activation Compression: Dynamic quantization only compresses model weights, leaving activations as FP32 until runtime where they are converted on the fly. Static quantization compresses both weights and activations ahead of time. This significantly reduces memory bandwidth requirements.

Hardware Tensor Core Acceleration: Jetson's integrated GPU features specialized Tensor Cores designed for fast INT8 matrix multiplication. These cores require fixed, static quantization parameters to operate at peak efficiency.

Jetson DLA (Deep Learning Accelerator) Support: If your Jetson model targets the hardware Deep Learning Accelerator (DLA) cores to save power, static quantization is strictly mandatory, as the DLA does not support dynamic operations.

## Transforms

In this implementation, each transformation of an image has a set chance to be applied on the image. Because TinyYoloV2 is a very small network, applying all transformations at once will lead to underfitting, as the network will struggle to make sense of the heavily transformed data. By setting the percent chance to a lower value like 0.3, we ensure that the network gets to learn transformed images, without them becoming to alien and impossible to learn.

### Rotations

During data augmentation, applying random rotations to training images causes severe bounding box inflation. Because object detection models require axis-aligned bounding boxes (AABBs), calculating the minimum and maximum coordinates of rotated box corners inherently expands their footprint, causing boxes to bloat and distort across the frame. Rotations are therefore limited to $\pm10^\circ$.