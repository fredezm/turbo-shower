import torch

# This is the most important check:
is_available = torch.cuda.is_available()
print(f"Is PyTorch able to use the GPU? {is_available}")

if is_available:
    # Get the number of available GPUs
    device_count = torch.cuda.device_count()
    print(f"Number of available GPUs: {device_count}")
    # Get the name of the current GPU
    current_device_name = torch.cuda.get_device_name(torch.cuda.current_device())
    print(f"Current GPU name: {current_device_name}")