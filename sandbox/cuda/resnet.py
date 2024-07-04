import os
import torch
import gc
from copy import deepcopy
from torch import nn, Tensor, device, no_grad, manual_seed
from torch import nn
from torchvision.datasets.utils import download_url
import torch.nn.functional as F
from torchmetrics.functional import accuracy
import torchvision
from torchvision import datasets, transforms
from torch.nn.functional import mse_loss

# Import functions defined in a specific path
t_PATH = os.path.abspath(__file__)
t_PATH = os.path.dirname(os.path.dirname(os.path.dirname(t_PATH)))
import sys
sys.path.append(t_PATH + '/src/')


from aihwkit.simulator.configs import ConstantStepDevice, SingleRPUConfig, FloatingPointDevice, FloatingPointRPUConfig
from aihwkit.optim import AnalogSGD
from aihwkit.inference.noise.base import BaseNoiseModel
from aihwkit.inference.noise.pcm import PCMLikeNoiseModel
from aihwkit.inference.compensation.drift import GlobalDriftCompensation
from aihwkit.inference.compensation.base import BaseDriftCompensation
from aihwkit.simulator.configs import InferenceRPUConfig
from aihwkit.simulator.presets.utils import PresetIOParameters
from aihwkit.simulator.parameters import (
    MappingParameter,
    IOParameters,
    PrePostProcessingParameter,
    InputRangeParameter,
    WeightClipParameter,
    WeightRemapParameter,
    WeightModifierParameter,
    WeightQuantizerParameter,
)
from aihwkit.simulator.parameters.enums import (
    WeightClipType,
    BoundManagementType,
    NoiseManagementType,
    WeightNoiseType,
    WeightRemapType,
    WeightModifierType,
)
# from aihwkit.utils.visualization import plot_device_compact
# from aihwkit.utils.analog_info import analog_summary
# from aihwkit.utils.fitting import fit_measurements
from aihwkit.nn.conversion import convert_to_analog
from aihwkit.utils.analog_info import analog_summary
from aihwkit.inference.calibration import (
    calibrate_input_ranges,
    InputRangeCalibrationType,
)
#from aihwkit.simulator.rpu_base import cuda

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from tqdm import tqdm

sys.path.append(t_PATH + '/sandbox/')

import src.plotting as pl
from src.utilities import import_mat_file

from src.noise import NullNoiseModel, ExperimentalNoiseModel
from aihwkit.inference.converter.conductance import SinglePairConductanceConverter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ********************************************************************************************************************
# ---------------------------------------------- MODEL DEFINITION ----------------------------------------------------
# ********************************************************************************************************************


class LambdaLayer(torch.nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)

# Definitions of some building blocks for the ResNet, taken form example '18_cifar10_on_resnet.ipynb'
class BasicBlock(torch.nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option="A"):
        super(BasicBlock, self).__init__()
        self.conv1 = torch.nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = torch.nn.BatchNorm2d(planes)
        self.conv2 = torch.nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = torch.nn.BatchNorm2d(planes)

        self.shortcut = torch.nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == "A":
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(
                    lambda x: F.pad(
                        x[:, :, ::2, ::2],
                        (0, 0, 0, 0, planes // 4, planes // 4),
                        "constant",
                        0,
                    )
                )
            elif option == "B":
                self.shortcut = torch.nn.Sequential(
                    torch.nn.Conv2d(
                        in_planes,
                        self.expansion * planes,
                        kernel_size=1,
                        stride=stride,
                        bias=False,
                    ),
                    torch.nn.BatchNorm2d(self.expansion * planes),
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Resnet9(torch.nn.Module):
    """
    From https://github.com/matthias-wright/cifar10-resnet/
    """

    def __init__(self, channels):
        super(Resnet9, self).__init__()

        self.channels = channels

        # resnet9 [56,112,224,224]
        # resnet9s [28,28,28,56]

        self.bn1 = torch.nn.BatchNorm2d(num_features=channels[0], momentum=0.9)
        self.bn2 = torch.nn.BatchNorm2d(num_features=channels[1], momentum=0.9)
        self.bn3 = torch.nn.BatchNorm2d(num_features=channels[2], momentum=0.9)
        self.bn4 = torch.nn.BatchNorm2d(num_features=channels[3], momentum=0.9)

        self.conv = torch.nn.Sequential(
            # prep
            torch.nn.Conv2d(
                in_channels=3,
                out_channels=channels[0],
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            self.bn1,
            torch.nn.ReLU(inplace=True),
            # Layer 1
            torch.nn.Conv2d(
                in_channels=channels[0],
                out_channels=channels[1],
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            self.bn2,
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            BasicBlock(in_planes=channels[1], planes=channels[1], stride=1),
            # Layer 2
            torch.nn.Conv2d(
                in_channels=channels[1],
                out_channels=channels[2],
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            self.bn3,
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            # Layer 3
            torch.nn.Conv2d(
                in_channels=channels[2],
                out_channels=channels[3],
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            self.bn4,
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            BasicBlock(in_planes=channels[3], planes=channels[3], stride=1),
            torch.nn.MaxPool2d(kernel_size=4, stride=4),
        )

        self.fc = torch.nn.Linear(in_features=channels[3], out_features=10, bias=True)

    def forward(self, x):
        out = self.conv(x)
        out = out.view(-1, self.channels[3])
        out = self.fc(out)
        return out

def resnet9s():
    return Resnet9(channels=[28, 28, 28, 56])

def get_test_loader(batch_size=32):
    transform_test = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
            ),
        ]
    )
    testset = torchvision.datasets.CIFAR10(
        root=t_PATH+"/sandbox/data/cifar10", train=False, download=True, transform=transform_test
    )
    test_loader = torch.utils.data.DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    return test_loader

def evaluate_model(model, test_loader, device):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for inputs, targets in tqdm(test_loader, desc="Evaluating model"):

            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        return 100.0 * correct / total
    

class Sampler:
    """Example of a sampler used for calibration."""

    def __init__(self, loader, device):
        self.device = device
        self.loader = iter(loader)
        self.idx = 0

    def __iter__(self):
        return self

    def __next__(self):
        x, _ = next(self.loader)
        self.idx += 1
        if self.idx > 100:
            raise StopIteration

        return ([x.to(self.device)], {})

# ********************************************************************************************************************
# ------------------------------------------- RPU CONFIG CUSTOMIZED --------------------------------------------------
# ********************************************************************************************************************


@dataclass 
class IdealPreset(InferenceRPUConfig):

    mapping: MappingParameter = field(
        default_factory=lambda: MappingParameter(
            weight_scaling_omega=1.0,
            weight_scaling_columnwise=False,
            max_input_size=512,
            max_output_size=0,
            digital_bias=True,
            learn_out_scaling=True,
            out_scaling_columnwise=True,
        )
    )

    # forward: IOParameters = field(
    #     default_factory= lambda:PresetIOParameters(
    #         is_perfect=True,
    #     )
    # )

    ''' *--------------------------------------------------------------------------------
    # OSS: As soon as the resolution for input and output is set to a value different from 0,
    # the input calibration step become mandatory
       --------------------------------------------------------------------------------'''
    forward: IOParameters = field(
        default_factory=lambda: PresetIOParameters(
            inp_res=254.0,
            out_res=254.0,
            bound_management=BoundManagementType.NONE,
            noise_management=NoiseManagementType.CONSTANT,
            nm_thres=1.0,
            # w_noise=0.0175,
            w_noise_type=WeightNoiseType.NONE,
            #ir_drop=1.0,
            # out_noise=0.04,
            # out_bound=10.0,
        )
    )

    backward: IOParameters = field(
        default_factory= lambda:PresetIOParameters(
            is_perfect=True,
        )
    )

    noise_model: BaseNoiseModel = field(default_factory=NullNoiseModel)

    pre_post: PrePostProcessingParameter = field(
        default_factory=lambda: PrePostProcessingParameter(
            # InputRangeParameter used for dynamic input range learning
            input_range=InputRangeParameter(
                enable=True,
                init_value=3.0,
                init_from_data=100,
                init_std_alpha=3.0,
                decay=0.001,
                input_min_percentage=0.95,
                output_min_percentage=0.95,
                manage_output_clipping=False,
                gradient_scale=1.0,
                gradient_relative=True,
            )
        )
    )






@dataclass
class CustomDefinedPreset(InferenceRPUConfig):

    mapping: MappingParameter = field(
        default_factory=lambda: MappingParameter(
            weight_scaling_omega=1.0,
            weight_scaling_columnwise=False,
            max_input_size=512,
            max_output_size=0,
            digital_bias=True,
            learn_out_scaling=True,
            out_scaling_columnwise=True,
        )
    )

    forward: IOParameters = field(
        default_factory=lambda: PresetIOParameters(
            inp_res=254.0,
            out_res=254.0,
            bound_management=BoundManagementType.NONE,
            noise_management=NoiseManagementType.CONSTANT,
            nm_thres=1.0,
            #w_noise=0.0175,
            w_noise_type=WeightNoiseType.PCM_READ,
            ir_drop=1.0,
            out_noise=0.04,
            out_bound=10.0,
        )
    )

    # remap: WeightRemapParameter = field(
    #     default_factory=lambda: WeightRemapParameter(
    #         remapped_wmax=1.0, type=WeightRemapType.CHANNELWISE_SYMMETRIC
    #     )
    # )


    # drift_compensation: Optional[BaseDriftCompensation] = field(
    #     default_factory=GlobalDriftCompensation
    # )

    pre_post: PrePostProcessingParameter = field(
        default_factory=lambda: PrePostProcessingParameter(
            # InputRangeParameter used for dynamic input range learning
            input_range=InputRangeParameter(
                enable=True,
                init_value=3.0,
                init_from_data=100,
                init_std_alpha=3.0,
                decay=0.001,
                input_min_percentage=0.95,
                output_min_percentage=0.95,
                manage_output_clipping=False,
                gradient_scale=1.0,
                gradient_relative=True,
            )
        )
    )

    # clip: WeightClipParameter = field(
    #     default_factory=lambda: WeightClipParameter(
    #         type=WeightClipType.FIXED_VALUE, fixed_value=1.0
    #     )
    # )

def get_quantized_model(model,level, rpu_config):
    rpu_config.quantization = WeightQuantizerParameter(
        resolution=0.12 if level == 17 else 0.2,
        levels=level
    )
    model_quantized = convert_to_analog(model, rpu_config)
    return model_quantized

# ********************************************************************************************************************
# ------------------------------------------- PLOTTING FUNCTIONS ------------------------------------------------------
# ********************************************************************************************************************

def accuracy_plot(model_names, inference_accuracy_values, observed_max, observed_min, r_number ,path):
    # Plot the accuracy of the models in a stem plot
    fig, ax = plt.subplots()
    y1= np.array([0.]*3)
    y2= np.array([0.]*3)
    for i, model_name in enumerate(model_names):
        mean = inference_accuracy_values[0, :, i].mean()
        std = inference_accuracy_values[0, :, i].std()
        y2[i] = mean + 3 * std
        y1[i] = mean - 3 * std
        ax.stem([model_name], [mean], linefmt="darkorange", markerfmt="D", basefmt=" ")
    ax.set_title(f"Accuracy of the models - n = {r_number} repeated measurements")
    ax.set_ylabel("Accuracy (%)")
    ax.set_xlim([-0.5, len(model_names) - 0.5])
    ax.minorticks_on()
    ax.yaxis.grid(True)
    ax.yaxis.grid(which="minor", linestyle=":", linewidth="0.5", color="gray")
    max = np.array(observed_max)
    min = np.array(observed_min)
    x = np.arange(len(model_names))
    ax.fill_between(x, y1, y2, where=(y2 > y1), color='bisque', alpha=0.5, label='Confidence Interval')
    ax.plot(x, y1, '--', color='firebrick')
    ax.plot(x, y2, '--', color = 'firebrick')
    ax.fill_between(x, min, max, where=(max > min), color='lightsalmon', alpha=0.5, label='Observed Accuracy Interval')
    ax.plot(x, max, ls='dashdot', color = 'olivedrab', label = 'Max observed accuracy', marker = '1', markersize=10)
    ax.plot(x, min, ls= 'dashdot', color = 'olivedrab', label = 'Min observed accuracy', marker = '2', markersize=10)
    ax.set_ylim([75, 82])
    ax.legend()

    # Save the plot to file
    plt.savefig(path)


# ********************************************************************************************************************
# ---------------------------------------------------- MAIN ----------------------------------------------------------
# ********************************************************************************************************************

if __name__ == '__main__':


    # -**-**-**-**-**-**-**-**-**-**-**-**-**-**-**- SETUP -**-**-**-**-**-**-**-**-**-**-**-**-**-**-**-

    p_PATH = os.path.abspath(__file__)
    p_PATH = os.path.dirname(os.path.dirname(p_PATH))

    # read the first argument, passed with the -l flag
    if len(sys.argv) > 1 and sys.argv[1] == '-l':
        if sys.argv[2] in ['9', '17']:
            SELECTED_LEVEL = int(sys.argv[2])
        else:
            raise Exception("Please specify a valid level of quantization (9 or 17)")
    else:
        raise Exception("Please specify the level of quantization with the -l flag")

    MAP = {
        9 : "matlab/3bit.mat",
        17 : "matlab/4bit.mat",
    }

    G_RANGE = [-40, 40]
    CONDUCTANCES = {
        9 : [G_RANGE[0] + i * (G_RANGE[1] - G_RANGE[0]) / 8 for i in range(9)],
        17 : [G_RANGE[0] + i * (G_RANGE[1] - G_RANGE[0]) / 16 for i in range(17)]
    }

    # Extract the data from the .mat file
    path = p_PATH+ f"/data/{MAP[SELECTED_LEVEL]}"
    variables = import_mat_file(path)

    types = variables['str']
    types = [types[0][t][0] for t in range(types.shape[1])]
    ww_mdn = variables['ww_mdn'] * 1e6
    ww_std = variables['ww_std'] * 1e6
    ww_mdn = pd.DataFrame(ww_mdn, columns=types)
    ww_std = pd.DataFrame(ww_std, columns=types)

    # Create the model and load the weights
    model = resnet9s().to(device)
    cwd = p_PATH
    os.mkdir(p_PATH+"/resnet") if not os.path.exists(p_PATH+"/resnet") else None
    os.mkdir(p_PATH+"/resnet/plots") if not os.path.exists(p_PATH+"/resnet/plots") else None
    if not os.path.exists(p_PATH+"/resnet/resnet9s.th"):
        download_url(
        "https://aihwkit-tutorial.s3.us-east.cloud-object-storage.appdomain.cloud/resnet9s.th",
        cwd + "/resnet/resnet9s.th",
        )
    state_dict = torch.load(p_PATH+"/resnet/resnet9s.th", device)
    # The state dict of the model with hardware-aware trained weights is stored in the
    # model_state_dict key of the external checkpoint.
    model.load_state_dict(state_dict["model_state_dict"], strict=True)
    rpu_config = IdealPreset()
    model = convert_to_analog(model, IdealPreset())
    model.eval()
    pl.generate_moving_hist(model,title="Distribution of Weight Values over the tiles - RESNET", file_name=p_PATH+"/resnet/plots/hist_resnet_UNQUATIZED.gif", range = (-.5,.5), top=None, split_by_rows=False, HIST_BINS = 171)

    model_quantized = get_quantized_model(model, SELECTED_LEVEL, rpu_config)
    model_quantized.eval()
    pl.generate_moving_hist(model_quantized,title= f"Distribution of Quantized Weight Values over the tiles - RESNET{SELECTED_LEVEL}", file_name=p_PATH+f"/resnet/plots/hist_resnet_QUANTIZED_{SELECTED_LEVEL}.gif", range = (-.5,.5), top=None, split_by_rows=False, HIST_BINS = 171)
    
    # -**-**-**-**-**-**-**-**-**-**-**-**-**-**-**- FIRST EVALUATION: 3 MODELS -**-**-**-**-**-**-**-**-**-**-**-**-**-**-**-

    t_inferences = [0.0]  # Times to perform infernece.
    n_reps = 10  # Number of inference repetitions.
    model_names = ["Unquantized", "Quantized - 9 levels", "Quantized - 17 levels"]
    inference_accuracy_values = torch.zeros((len(t_inferences), n_reps, len(model_names)))
    observed_max = [0] * len(model_names)
    observed_min = [100] * len(model_names)
    for i,model_name in enumerate(model_names):
        for t_id, t in enumerate(t_inferences):
            for j in range(n_reps):
            # For each repetition, get a new version of the quantized model and calibrare it

                if model_name == "Unquantized":
                    model_i = deepcopy(model)
                else:
                    model_i = get_quantized_model(model, SELECTED_LEVEL, rpu_config)
                model_i.eval()
                
                # Calibrate input ranges
                dataloader=Sampler(get_test_loader(), device)
                calibrate_input_ranges(
                model=model_i,
                calibration_type=InputRangeCalibrationType.CACHE_QUANTILE,
                dataloader=dataloader,
                )
                
                # Compute the accuracies
                inference_accuracy_values[t_id, j, i] = evaluate_model(
                    model_i, get_test_loader(), device
                )
                if observed_max[i] < inference_accuracy_values[t_id, j, i]:
                    observed_max[i] = inference_accuracy_values[t_id, j, i]
                if observed_min[i] > inference_accuracy_values[t_id, j, i]:
                    observed_min[i] = inference_accuracy_values[t_id, j, i]
                print(f"Accuracy on rep:{j}, model:{i} -->" , inference_accuracy_values[t_id, j, i])
                # tile_weights = next(model_i.analog_tiles()).get_weights()
                # print(f"Tile weights for model {model_names[i]}: {tile_weights[0][0:5, 0:5]}")
                
                del model_i
                del dataloader
                torch.cuda.empty_cache()
                gc.collect()
                #torch.cuda.reset_peak_memory_stats()

        print(
                f"Test set accuracy (%) at t={t}s for {model_names[i]}: mean: {inference_accuracy_values[t_id, :, i].mean()}, std: {inference_accuracy_values[t_id, :, i].std()}"
            )
            

    accuracy_plot(model_names, inference_accuracy_values, observed_max, observed_min, n_reps ,path=p_PATH + "/resnet/plots/accuracy_resnet.png")


    # -**-**-**-**-**-**-**-**-**-**-**-**-**-**-**- SECOND EVALUATION: FITTED DATA -**-**-**-**-**-**-**-**-**-**-**-**-**-**-**-
    print("Available experimental noises are: ", types)
    CHOSEN_NOISE = types[0]
    print(f"Chosen noise: {CHOSEN_NOISE}" )
    path = p_PATH + f"/data/{MAP[SELECTED_LEVEL]}"
    print(f"Selected level: {SELECTED_LEVEL}")

    RPU_CONFIG  = IdealPreset()
    RPU_CONFIG.noise_model= ExperimentalNoiseModel(file_path = path,
                                                type = CHOSEN_NOISE,
                                                g_converter=SinglePairConductanceConverter(g_max=40.)),
                    

    original_model = resnet9s().to(device)
    original_model.load_state_dict(state_dict["model_state_dict"], strict=True)

    '''QUANTIZED 9 levels'''
    RPU_CONFIG.quantization = WeightQuantizerParameter(
        resolution=0.2 if SELECTED_LEVEL == 9 else 0.12,
        levels = SELECTED_LEVEL,
    )
    model_fitted = convert_to_analog(original_model, RPU_CONFIG)
    model_fitted.eval()
    tile_weights = next(model_fitted.analog_tiles()).get_weights()
    pl.plot_tensor_values(tile_weights[0], 141, (-.6,.6), f"Distribution of quantized weights - Conv1 - RESNET{SELECTED_LEVEL}", p_PATH + f"/resnet/plots/hist_resnet_QUANTIZED_{SELECTED_LEVEL}_Conv1.png")
    weight_max = max(abs(tile_weights[0].flatten().numpy()))
    model_fitted.program_analog_weights()


    # Plot the histogram of the weights of the last model
    tile_weights = next(model_fitted.analog_tiles()).get_weights()
    gaussain_noise = {"means": ww_mdn[CHOSEN_NOISE].values, "stds": ww_std[CHOSEN_NOISE].values, "gmax": 40.0}
    pl.plot_tensor_values(tile_weights[0], 141, (-.6,.6), f"Distribution of quantized weights + Fitted Noise ({CHOSEN_NOISE}) - Conv1 - RESNET{SELECTED_LEVEL}", p_PATH + f"/resnet/plots/hist_resnet_QUANTIZED_{SELECTED_LEVEL}+FITTED_Conv1.png")
    pl.plot_tensor_values(tile_weights[0], 141, (-.6,.6), f"Distribution of quantized weights + Fitted Noise ({CHOSEN_NOISE}) - Conv1+Gaussian \n- RESNET{SELECTED_LEVEL}", p_PATH + f"/resnet/plots/hist_resnet_QUANTIZED_{SELECTED_LEVEL}+FITTED_Conv1+Gaussian.png", gaussian=gaussain_noise, weight_max=weight_max)
    pl.generate_moving_hist(model_fitted,title=f"Distribution of Quantized Weight + Fitted Noise ({CHOSEN_NOISE})\n Values over the tiles - RESNET{SELECTED_LEVEL}", file_name=p_PATH + f"/resnet/plots/hist_resnet_QUANTIZED_{SELECTED_LEVEL}_FITTED.gif", range = (-.5,.5), top=None, split_by_rows=False, HIST_BINS=171)

    # Estimate the accuracy of the model with the fitted noise with respect to the other 9 levels model
    fitted_models_names = []
    fitted_models_accuracy = torch.zeros((len(t_inferences), n_reps, len(types)))
    fitted_observed_max = [0] * len(types)
    fitted_observed_min = [100] * len(types)
    for i in range(len(types)):
        CHOSEN_NOISE = types[i]
        RPU_CONFIG  = IdealPreset()
        RPU_CONFIG.quantization = WeightQuantizerParameter(
            resolution=0.2 if SELECTED_LEVEL == 9 else 0.12,
            levels = SELECTED_LEVEL,
            )
        RPU_CONFIG.noise_model=ExperimentalNoiseModel(file_path = path,
                                                        type = CHOSEN_NOISE,
                                                        g_converter=SinglePairConductanceConverter(g_max=40.)),

        fitted_models_names.append(f"Quantized - {SELECTED_LEVEL} levels \n+ Fitted Noise \n ({CHOSEN_NOISE})")
        for t_id, t in enumerate(t_inferences):
            for j in range(n_reps):
                # For each repetition, get a new version of the quantized model and calibrare it
                model_fitted = convert_to_analog(original_model, RPU_CONFIG)
                model_fitted.eval()
                model_fitted.program_analog_weights()

                # Calibrate input ranges
                dataloader = Sampler(get_test_loader(), device)
                calibrate_input_ranges(
                model=model_fitted,
                calibration_type=InputRangeCalibrationType.CACHE_QUANTILE,
                dataloader=dataloader,
                )

                # Then evaluate the model
                fitted_models_accuracy[t_id, j, i] = evaluate_model(model_fitted, get_test_loader(), device)
                if fitted_observed_max[i] < fitted_models_accuracy[t_id, j, i]:
                    fitted_observed_max[i] = fitted_models_accuracy[t_id, j, i]
                if fitted_observed_min[i] > fitted_models_accuracy[t_id, j, i]:
                    fitted_observed_min[i] = fitted_models_accuracy[t_id, j, i]
                
                # Delete the model to free CUDA memory
                del model_fitted
                del dataloader
                torch.cuda.empty_cache()
                gc.collect()
                #torch.cuda.reset_peak_memory_stats()
            print(
                f"Test set accuracy (%) at t={t}s for {fitted_models_names[i]}: mean: {fitted_models_accuracy[t_id, :, i].mean()}, std: {fitted_models_accuracy[t_id, :, i].std()}"
            )

    # Plot the accuracy of the models in a stem plot
    fig, ax = plt.subplots(figsize=(23,7))
    models = ["Unquantized",f"Quantized - {SELECTED_LEVEL} levels"] + fitted_models_names
    if SELECTED_LEVEL == 9:
        accuracies = [inference_accuracy_values[t_id, :, 0].mean(),inference_accuracy_values[t_id, :, 1].mean()]
        std_accuracy = [inference_accuracy_values[t_id, :, 0].std(),inference_accuracy_values[t_id, :, 1].std()]
        observed_max = observed_max[:2]
        observed_min = observed_min[:2]
    else:
        accuracies = [inference_accuracy_values[t_id, :, 0].mean(),inference_accuracy_values[t_id, :, 2].mean()]
        std_accuracy = [inference_accuracy_values[t_id, :, 0].std(),inference_accuracy_values[t_id, :, 2].std()]
        observed_max = [observed_max[0], observed_max[2]]
        observed_min = [observed_min[0], observed_min[2]]
    accuracies = accuracies + fitted_models_accuracy.mean(dim=1)[0].tolist()
    std_accuracy = std_accuracy + fitted_models_accuracy.std(dim=1)[0].tolist()
    observed_max = observed_max + fitted_observed_max
    observed_min = observed_min + fitted_observed_min
    ax.stem(models[:2], accuracies[:2], linefmt ='darkorange', markerfmt ='D', basefmt=' ')
    ax.stem(models[2:], accuracies[2:], linefmt ='darkorchid', markerfmt ='D', basefmt=' ')
    # Define the points for the boundary lines
    x = np.arange(len(models))
    y1 = np.array([accuracies[i] - 3*std_accuracy[i] for i in range(len(models))])
    y2 = np.array([accuracies[i] + 3*std_accuracy[i] for i in range(len(models))])
    max = np.array(observed_max)
    min = np.array(observed_min)
    # Interpolating or directly using the points to fill the region
    ax.fill_between(x, y1, y2, where=(y2 > y1), color='bisque', alpha=0.5, label='Confidence Interval')
    ax.plot(x, y1, '--', color='firebrick')
    ax.plot(x, y2, '--', color = 'firebrick')
    ax.fill_between(x, min, max, where=(max > min), color='lightsalmon', alpha=0.5, label='Observed Accuracy Interval')
    ax.plot(x, max, ls='dashdot', color = 'olivedrab', label = 'Max observed accuracy', marker = '1', markersize=10)
    ax.plot(x, min, ls= 'dashdot', color = 'olivedrab', label = 'Min observed accuracy', marker = '2', markersize=10)


    ax.set_title(f"Accuracy of the models over {n_reps} repetitions")
    ax.set_ylabel("Accuracy (%)")
    ax.set_xlim([-0.5, len(models)- 0.5])
    ax.minorticks_on()
    ax.yaxis.grid(True)
    ax.yaxis.grid(which='minor', linestyle=':', linewidth='0.5', color='gray')
    ax.set_ylim([50, 90])
    ax.legend()
    # Save the plot to file
    plt.savefig(p_PATH+f"/resnet/plots/accuracy_resnet_FittedNoise_{SELECTED_LEVEL}.png")


