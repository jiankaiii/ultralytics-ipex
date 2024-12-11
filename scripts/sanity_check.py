import torch
import intel_extension_for_pytorch as ipex

print("Checking PyTorch version:")
print(torch.__version__)  # Check PyTorch version

print("\nChecking IPEX version:")
print(f"{ipex.__version__}\n")

# Check if XPU (Intel GPUs) is available
if torch.xpu.is_available():
    # Loop over the devices and print their properties
    for i in range(torch.xpu.device_count()):
        device_properties = torch.xpu.get_device_properties(i)
        print(f"[{i}]: {device_properties}")
else:
    print(
        "No XPU devices available. Please ensure you have the Intel XPU (Level Zero) drivers installed."
    )

