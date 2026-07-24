# Setup

This project is managed using **uv**.
You need to first install **uv** to be able to initialize the python environment.

After installing uv execute

```
uv sync
```

After that you can source the venv as usual and execute the scripts.



# General notes

## Efficiency Score

To select the optimal model after pruning, we must evaluate both its object detection accuracy and its computational speedup. We use Average Precision (AP) rather than Validation/Test Loss as our primary accuracy metric.

### Why We Use AP Instead of Loss
Loss is a continuous proxy metric that calculates exact mathematical error (like Cross-Entropy). While necessary for backpropagation, it does not perfectly reflect final object detection quality. When a network is pruned by removing parameters, its overall confidence scores often drop slightly. This drop causes the cross-entropy loss to spike, making the model appear heavily degraded. 

However, as long as those slightly lower confidence scores remain above our NMS and filtering thresholds (e.g., a box dropping from 90% to 65% confidence against a 25% threshold), the final bounding boxes output by the model remain completely unchanged. Therefore, Average Precision (AP), which evaluates the final, thresholded predictions, provides a much more accurate representation of the true performance impact of pruning.

To find the best performing model that balances accuracy retention and computational speed, we use the following equation:

$$
\text{Efficiency Score} = \text{AP}_{\text{current}} \times \left( \frac{\text{FPS}_{\text{current}}}{\text{FPS}_{\text{baseline}}} \right)
$$

This calculates a speed-adjusted AP score, allowing us to mathematically quantify whether a slight drop in precision is worth the frame-rate gain. Because we are maximizing both accuracy and speed, **a higher score is better**.

## ONNX Quantization

We use static quantization for the following reasons:

TensorRT Engine Compatibility: The TensorrtExecutionProvider on Jetson cannot natively fuse or optimize dynamically quantized models. Static quantization (using QDQ nodes) creates a graph structure that TensorRT can fully parse, compile, and optimize into a high-performance execution engine.

Elimination of Runtime Overhead: Dynamic quantization computes activation scales on the fly during inference, introducing extra computational steps. Static quantization bakes these scaling factors directly into the graph, saving valuable GPU cycles on the edge.

True Activation Compression: Dynamic quantization only compresses model weights, leaving activations as FP32 until runtime where they are converted on the fly. Static quantization compresses both weights and activations ahead of time. This significantly reduces memory bandwidth requirements.

Hardware Tensor Core Acceleration: Jetson's integrated GPU features specialized Tensor Cores designed for fast INT8 matrix multiplication. These cores require fixed, static quantization parameters to operate at peak efficiency.

Jetson DLA (Deep Learning Accelerator) Support: If your Jetson model targets the hardware Deep Learning Accelerator (DLA) cores to save power, static quantization is strictly mandatory, as the DLA does not support dynamic operations.

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