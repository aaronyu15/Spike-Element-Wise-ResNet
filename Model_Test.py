import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.clock_driven.neuron import MultiStepParametricLIFNode
from spikingjelly.clock_driven import layer
from spikingjelly.datasets import dvs128_gesture
import custom_spikingjelly_class
import os
import utilities
from utilities import Q3_5QuantizedTensor, Q3_5Hook
import dvsgesture.smodels as smodels
import argparse
from Model import ResNetN
from torchvision import transforms

def load_and_quantize_weights(model, weights_path=None, use_coe=False):
    """
    Load pre-trained weights and quantize them to Q3.5.
    If weights_path is None, just quantize the initiated random weights.
    If use_coe is True, load weights from COE files. This more resembles the hardware implementation.
    """

    if(use_coe):
        conv_data    = utilities.read_coe(conv_coe, in_bin=True)

        fc_weight = []

        for i in range(11):
            fc_weight.append(utilities.read_coe(fc_data_coe[i], in_bin=True))

        conv_data_weight = torch.tensor(conv_data[0:-6])
        conv_data_bias = torch.tensor(conv_data[-6:])
        fc_weight = torch.tensor(fc_weight)

        conv_data_weight = conv_data_weight.reshape(6,2,3,3)
        model.conv2d_1[0].weight.data = conv_data_weight
        model.conv2d_1[0].bias.data = conv_data_bias  

        model.out.weight.data = fc_weight

    elif weights_path is not None:
        # Load pre-trained weights
        checkpoint = torch.load(weights_path, weights_only=False)
        model.load_state_dict(checkpoint['model'])

    
    # Quantize all parameters to Q3.5
    model.quantize_parameters()
    
    return model

def transform_data(data, args=None):
    """
    Transform the data with some noise or other effects.
    Data is a tensor of shape (N, T, C, H, W). 
    """
    # Apply random transformations to each time step independently
    N, T, C, H, W = data.shape

    # Define custom transform for camera motion (e.g., random affine transformations)
    camera_motion_transform = transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1))

    transformed_data = []
    for t in range(T):  # Iterate over the time dimension
        time_step_data = data[:, t]  # Extract data for the current time step

        transformed_time_step = time_step_data  # Initialize with the original data

        # Apply the camera motion transform to the time step
        if args.camera_movement:
            transformed_time_step = camera_motion_transform(time_step_data)  # Apply the transform

        if args.sp_noise:
            # Apply salt and pepper noise to the transformed time step
            transformed_time_step = utilities.SaltPepperNoise(prob=args.sp_noise_prob)(transformed_time_step)

        transformed_data.append(transformed_time_step)

    # Stack the transformed time steps back into a single tensor
    data = torch.stack(transformed_data, dim=1)

    return data


def test_quantized_network(args):
    """Test the quantized network with example data"""
    model = ResNetN(args=args)
    
    # Load and quantize weights
    model = load_and_quantize_weights(model, weights_path=model_params, use_coe=args.use_coe)
    model = model.to(device)

    # Make sure hooks are applied
    model.apply_quantization_hooks()


    # Initialize variables to track accuracy
    correct = 0
    top3_correct = 0

    total = 0

    eval_loader = test_loader if args.eval_data == "test" else data_loader

    # Disable gradient computation for evaluation
    with torch.no_grad():
        for data, labels in eval_loader:
            # Move data and labels to the same device as the model
            data, labels = data.to(device), labels.to(device)

            # data_preprocess() performs the preprocessing of the data (normalization, maxpooling, quantization)
            data = transform_data(data, args=args)

            data = utilities.data_preprocess(data, bin_rep=False, args=args)

            ## Forward pass through the model
            outputs = model(data)

            # Get the predicted class (highest score)
            _, predicted = torch.max(outputs, 1)
            
            # Update total and correct counts
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            top3_correct += torch.sum(outputs.topk(3, dim=1)[1] == labels.view(-1, 1)).item()

            if args.save_gif:
                utilities.play(data[0], save_gif=True, file_name=f"output_images/{args.gif_name}")
                break



    # Calculate and print accuracy
    accuracy = 100 * correct / total
    top3_accuracy = 100 * top3_correct / total
    print(f"Top-1 Accuracy on the {args.eval_data} dataset: {accuracy:.2f}%") 
    print(f"Top-3 Accuracy on the {args.eval_data} dataset: {top3_accuracy:.2f}%") 




if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Test the quantized ResNetN network.")
    parser.add_argument("--use_coe", action="store_true", default=False, help="Use COE files for weights instead of a checkpoint.")
    parser.add_argument("--eval_data", type=str, default="test", choices=["train", "test"], help="Evaluation data, either 'train' or 'test'.")
    parser.add_argument("--train", action="store_true", default=False, help="Whether or not a model is being trained.")
    parser.add_argument("--camera_movement", action="store_true", default=False, help="Add camera movement to the data.")
    parser.add_argument("--sp_noise", action="store_true", default=False, help="Add salt and pepper noise to the data.")
    parser.add_argument("--sp_noise_prob", type=float, default=0.05, help="Probability of salt and pepper noise.")
    parser.add_argument("--save_gif", action="store_true", default=False, help="Save the output as a GIF.")
    parser.add_argument("--gif_name", type=str, default="gesture.gif", help="Save the output as a GIF.")
    args = parser.parse_args()

    """
    Common statements
    """

    device = torch.device("cuda")
    device = torch.device("cpu")
    if(args.train):
        seed = 19
        generator = torch.Generator().manual_seed(seed)
        torch.set_grad_enabled(False)
    else:
        generator = torch.Generator()

    torch.set_printoptions(threshold=float("inf"), precision=10)

    # Dataset Directory
    DVSGesture_dir = "../DvsGesture/"
    root_dir  = os.path.join(DVSGesture_dir, "events_np/train/0")
    frame_dir = os.path.join(DVSGesture_dir, "frames_number_16_split_by_number/train/0")

    # Working Directory
    working_dir = "."
    checkpoint_path = os.path.join(working_dir, "dvsgesture/logs/26_no_bias/lr0.001")
    checkpoint_file = "checkpoint_299.pth"

    model_params = os.path.join(checkpoint_path, checkpoint_file)


    # Apply the transforms to the dataset
    dataset_train = dvs128_gesture.DVS128Gesture(
        root=DVSGesture_dir, train=True, data_type='frame', frames_number=16, split_by='number'
    )
    sampler_train = torch.utils.data.RandomSampler(dataset_train, generator=generator)

    dataset_test  = dvs128_gesture.DVS128Gesture(
        root=DVSGesture_dir, train=False, data_type='frame', frames_number=16, split_by='number'
    )
    sampler_test = torch.utils.data.RandomSampler(dataset_test, generator=generator)

    data_loader = torch.utils.data.DataLoader(
        dataset_train, batch_size=49,
        sampler=sampler_train, num_workers=0, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(
        dataset_test, batch_size=32,
        sampler=sampler_test, num_workers=0, pin_memory=True)

    # coe file related
    test_data_coe = "./coe_files/test_data.coe"
    conv_coe = "./coe_files/conv3x3.coe"
    fc_data_coe = [f"./coe_files/fc{i}.coe" for i in range(11)]

    output = test_quantized_network(args)