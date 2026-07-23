import torch
from tinyyolov2_fused import TinyYoloV2Fused
from tinyyolov2 import TinyYoloV2
from utils.dataloader import VOCDataLoader
import time 

def fuse_conv_bn_weights(conv_w, conv_b, bn_rm, bn_rv, bn_w, bn_b):
    """
    Fuses the convolutional weights and biases with the batchnorm weights and biases.
    Input:
        conv_w:          shape=(output_channels, in_channels, kernel_size, kernel_size)
        conv_b:          shape=(output_channels)
        bn_rm:           shape=(output_channels)
        bn_rv(sigma^2):  shape=(output_channels)
        bn_w:            shape=(output_channels)
        bn_b:            shape=(output_channels)
    
    Output:
        fused_conv_w = shape=conv_w
        fused_conv_b = shape=conv_b
    """
    bn_eps = 1e-05

    fused_conv = torch.zeros(conv_w.shape)
    fused_bias = torch.zeros(conv_b.shape)
    
    shape = [-1] + [1] * (len(conv_w.shape) - 1) #change 1d vector to [max, 1, 1, 1, ...], (max is here 16/32/64)
    fused_conv = conv_w * torch.div(bn_w,(torch.sqrt(bn_rv + bn_eps))).reshape(shape)
    fused_bias = (bn_w * (conv_b - bn_rm)) / torch.sqrt(bn_rv + bn_eps) + bn_b

    return fused_conv, fused_bias

def fuse_conv_bn_weights_only(conv_w, bn_rv, bn_w):
    """
    Fuses only the weights of the convolutional and batchnorm layers.
    Input:
        conv_w:          shape=(output_channels, in_channels, kernel_size, kernel_size)
        bn_rv(sigma^2):  shape=(output_channels)
        bn_w:            shape=(output_channels)
    
    Output:
        fused_conv_w = shape=conv_w
    """
    bn_eps = 1e-05

    fused_conv = torch.zeros(conv_w.shape)
    
    shape = [-1] + [1] * (len(conv_w.shape) - 1) #change 1d vector to [max, 1, 1, 1, ...], (max is here 16/32/64)
    fused_conv = conv_w * torch.div(bn_w,(torch.sqrt(bn_rv + bn_eps))).reshape(shape)

    return fused_conv


def net_time(model_class, testloader):
    """
    Calculates the mean time needed for inference for a single batch on the CPU
    """
    net = model_class()
    #net.load_state_dict(torch.load("state_dict.pt", weights_only=True)) 
    net.eval()
    batch = 0
    t_start = time.time()
    for _, (inputs, targets) in enumerate(testloader):
        output = net(inputs)
        batch += 1
        if(batch == 100):
            break
    t_end = time.time()
    t = (t_end - t_start) / batch
    return t

def net_acc(model_class, state_dict, testloader):
    """
    Calculates the mean accuracy of a batch.
    """
    num_correct = 0
    num_samples = 0
    batch = 0
    
    net = model_class()
    net.load_state_dict(state_dict) 
    net.eval()
    for _, (inputs, targets) in enumerate(testloader):
        #inputs, targets = inputs.to(device), targets.to(device)
        outputs = net(inputs)
        num_correct += (torch.argmax(outputs, dim=1) == targets).sum().item()
        batch += 1
        if(batch == 100):
            break
    num_samples = testloader.batch_size * batch
    accuracy = num_correct / num_samples
    return accuracy

def fuse_net(base_model_class, base_state_dict, fused_model_class, fused_state_dict_path, print_time=False):
    """
    Fuses the conv and batchnorm layers of a base model and returns the fused state dict.
    """
    fused_net = fused_model_class()
    fused_net_state_dict = fused_net.state_dict()
    #for key in fused_net_state_dict: print(key, fused_net_state_dict[key].dtype)

    saved_state_dict = torch.load(base_state_dict, weights_only=True)

    num_conv_layers = 9
    for i in range(1, num_conv_layers):
        conv_weight = saved_state_dict[f"conv{i}.weight"]
        bn_weight = saved_state_dict[f"bn{i}.weight"]
        #bn_bias = saved_state_dict[f"bn{i}.bias"]
        #bn_running_mean = saved_state_dict[f"bn{i}.running_mean"]
        bn_running_var = saved_state_dict[f"bn{i}.running_var"]
        fused_weight = fuse_conv_bn_weights_only(conv_weight, bn_running_var, bn_weight)
        fused_net_state_dict[f"conv{i}.weight"] = fused_weight

    # set the last layer extra because it has a bias
    fused_net_state_dict[f"conv{num_conv_layers}.weight"] = saved_state_dict[f"conv{num_conv_layers}.weight"]
    fused_net_state_dict[f"conv{num_conv_layers}.bias"] = saved_state_dict[f"conv{num_conv_layers}.bias"]

    torch.save(fused_net_state_dict, fused_state_dict_path)
    if print_time:
        loader = VOCDataLoader(split=False, batch_size=1)
        print(f'Time unfused: {net_time(base_model_class, loader)} s')
        print(f'Time fused: {net_time(fused_model_class, loader)} s')
    return fused_net_state_dict


fused_net = TinyYoloV2Fused()
fused_net_state_dict = fused_net.state_dict()
#for key in fused_net_state_dict: print(key, fused_net_state_dict[key].dtype)

saved_state_dict = torch.load('state_dicts/voc_pretrained.pt', weights_only=True)

num_conv_layers = 9
for i in range(1, num_conv_layers):
    conv_weight = saved_state_dict[f"conv{i}.weight"]
    bn_weight = saved_state_dict[f"bn{i}.weight"]
    bn_bias = saved_state_dict[f"bn{i}.bias"]
    bn_running_mean = saved_state_dict[f"bn{i}.running_mean"]
    bn_running_var = saved_state_dict[f"bn{i}.running_var"]
    fused_weight = fuse_conv_bn_weights_only(conv_weight, bn_running_var, bn_weight)
    fused_net_state_dict[f"conv{i}.weight"] = fused_weight

# set the last layer extra because only this layer has a bias
fused_net_state_dict[f"conv{num_conv_layers}.weight"] = saved_state_dict[f"conv{num_conv_layers}.weight"]
fused_net_state_dict[f"conv{num_conv_layers}.bias"] = saved_state_dict[f"conv{num_conv_layers}.bias"]

torch.save(fused_net_state_dict, "state_dicts/yolov2_fused.pt")

loader = VOCDataLoader(split=False, batch_size=1)
print(f'Time unfused: {net_time(TinyYoloV2, loader)} s')
#print(f"Accuracy unfused: {net_acc(TinyYoloV2, torch.load('voc_pretrained.pt'), loader):.4%}")
print(f'Time fused: {net_time(TinyYoloV2Fused, loader)} s')
#print(f"Accuracy fused: {net_acc(TinyYoloV2Fused, net_state_dict, loader):.4%}")
