import torch
import torch.nn as nn
from torch.quantization import QuantStub, DeQuantStub
import onnx
from dvsgesture.smodels import ResNetN
import os


# Function to quantize input data
def quantize_input(input_tensor, scale, zero_point):
    """
    Quantize input tensor to int8 using scale and zero-point.
    input_tensor: Float tensor (e.g., normalized to [0, 1])
    scale: Quantization scale (float)
    zero_point: Quantization zero-point (int)
    Returns: int8 tensor
    """
    # Quantize: q = round(x / scale + zero_point)
    quantized = torch.round(input_tensor / scale + zero_point).to(torch.int8)
    return quantized

# Main quantization process
def quantize_model_and_export():
    # Instantiate and load trained model
    model = ResNetN().eval()
    # Assume model is trained; load weights if needed

    working_dir = "/home/aaron/Research_Projs/Spike-Element-Wise-ResNet/"
    checkpoint_path = os.path.join(working_dir, "dvsgesture/logs/26_no_bias/lr0.001")
    checkpoint_file = "checkpoint_299.pth"

    model_weights = torch.load(os.path.join(checkpoint_path, checkpoint_file))

    model.load_state_dict(torch.load(model_weights))

    # Fuse layers (e.g., Conv + ReLU)
    model = torch.quantization.fuse_modules(
        model, [['conv1', 'relu1'], ['conv2', 'relu2']], inplace=True
    )

    # Set quantization configuration
    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')  # For calibration
    torch.backends.quantized.engine = 'fbgemm'

    # Prepare model for quantization
    torch.quantization.prepare(model, inplace=True)

    # Calibrate with representative dataset
    calibration_data = torch.randn(100, 1, 28, 28)  # Simulate MNIST-like data
    with torch.no_grad():
        for data in calibration_data.split(1):
            model(data)

    # Convert to quantized model
    quantized_model = torch.quantization.convert(model, inplace=False)

    # Extract quantization parameters
    input_scale = quantized_model.quant.scale.item()
    input_zero_point = quantized_model.quant.zero_point.item()
    print(f"Input scale: {input_scale}, Input zero-point: {input_zero_point}")

    # Extract and print weight quantization parameters
    for name, param in quantized_model.named_parameters():
        if hasattr(param, 'q_per_channel_scales'):
            scales = param.q_per_channel_scales()
            zero_points = param.q_per_channel_zero_points()
            print(f"{name} scales: {scales}, zero-points: {zero_points}")
        else:
            scale = param.q_scale()
            zero_point = param.q_zero_point()
            print(f"{name} scale: {scale}, zero-point: {zero_point}")

    # Save quantized model
    torch.save(quantized_model.state_dict(), "quantized_model.pth")

    # Export to ONNX
    dummy_input = torch.randn(1, 1, 28, 28)
    torch.onnx.export(
        quantized_model,
        dummy_input,
        "quantized_model.onnx",
        opset_version=13,
        input_names=['input'],
        output_names=['output']
    )
    print("Exported quantized model to quantized_model.onnx")

    # Example: Quantize input data
    sample_input = torch.randn(1, 1, 28, 28)  # Replace with real input
    quantized_input = quantize_input(sample_input, input_scale, input_zero_point)
    print(f"Quantized input (int8): {quantized_input}")

    return quantized_model, input_scale, input_zero_point

# Run quantization
if __name__ == "__main__":
    quantized_model, input_scale, input_zero_point = quantize_model_and_export()