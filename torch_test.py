import torch

# Check if PyTorch is installed
print("PyTorch Version:", torch.__version__)

# Test matrix creation
x = torch.rand(5, 3)
print("Random Tensor:\n", x)

# Check if CUDA (GPU) is active and available
print("CUDA Available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("Device Name:", torch.cuda.get_device_name(0))
    print("Device Count:", torch.cuda.device_count())