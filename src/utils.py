import os
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from torch.utils.data import Dataset, TensorDataset, ConcatDataset

logger = logging.getLogger(__name__)


#######################
# TensorBaord setting #
#######################
def launch_tensor_board(log_path, port, host):
    """Function for initiating TensorBoard.
    
    Args:
        log_path: Path where the log is stored.
        port: Port number used for launching TensorBoard.
        host: Address used for launching TensorBoard.
    """
    os.system(f"tensorboard --logdir={log_path} --port={port} --host={host}")
    return True

#########################
# Weight initialization #
#########################
def init_weights(model, init_type, init_gain):
    """Function for initializing network weights.
    
    Args:
        model: A torch.nn instance to be initialized.
        init_type: Name of an initialization method (normal | xavier | kaiming | orthogonal).
        init_gain: Scaling factor for (normal | xavier | orthogonal).
    
    Reference:
        https://github.com/DS3Lab/forest-prediction/blob/master/pix2pix/models/networks.py
    """
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            else:
                raise NotImplementedError(f'[ERROR] ...initialization method [{init_type}] is not implemented!')
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        
        elif classname.find('BatchNorm2d') != -1 or classname.find('InstanceNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)   
    model.apply(init_func)

def init_net(model, init_type, init_gain, gpu_ids):
    """Function for initializing network weights.
    
    Args:
        model: A torch.nn.Module to be initialized
        init_type: Name of an initialization method (normal | xavier | kaiming | orthogonal)l
        init_gain: Scaling factor for (normal | xavier | orthogonal).
        gpu_ids: List or int indicating which GPU(s) the network runs on. (e.g., [0, 1, 2], 0)
    
    Returns:
        An initialized torch.nn.Module instance.
    """
    # if len(gpu_ids) > 0:
    #     assert(torch.cuda.is_available())
    #     model.to(gpu_ids[0])
    #     model = nn.DataParallel(model, gpu_ids)
    init_weights(model, init_type, init_gain)
    return model

#################
# Dataset split #
#################
class CustomTensorDataset(Dataset):
    """TensorDataset with support of transforms."""
    def __init__(self, tensors):
        # assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors

    def __getitem__(self, index):
        X = self.tensors[0][index] 
        y = self.tensors[1][index]
        return X, y

    def __len__(self):
        return self.tensors[0].size(0)

def preprocessing_training_dataset(dataset, dataset_name):
    # remove any inf or nan values
    dataset = dataset.replace([np.inf, -np.inf], np.nan)
    dataset = dataset.dropna()

    # drop irrelevant features 
    if dataset_name in ["CIC-ToN-IoT"]:
        dataset = dataset.drop(["Flow ID", "Src IP", "Dst IP", "Timestamp", "Attack", 'CWE Flag Count', 'ECE Flag Cnt'], axis=1)
    column_names = np.array(dataset.columns)
    to_drop = []
    for x in column_names:
        size = dataset.groupby([x]).size()
        #check for cols that only take an unique value
        if (len(size.unique()) == 1):
            to_drop.append(x)
    # print(to_drop)
    dataset = dataset.drop(to_drop, axis=1)
    print("Dataset after processing:", dataset.shape)
    # X,y of data
    trainx = dataset.iloc[:,:dataset.shape[1]-1]
    trainy = dataset.iloc[:,dataset.shape[1]-1]

    # Normalize data
    mms = MinMaxScaler().fit(trainx)
    trainx = mms.transform(trainx)
    trainx = np.reshape(trainx, (trainx.shape[0], 1, trainx.shape[1]))
    return trainx, trainy


def create_datasets(dataset_name, num_clients, iid, attack_mode):
    """Split the whole dataset in IID or non-IID manner for distributing to clients."""
    # get dataset 
    if dataset_name in ["CIC-ToN-IoT"]:
        full_dataset = pd.read_csv("/home/haochu/Documents/PoisoningAttack/Dataset/CICToNIoT/Original/CIC-ToN-IoT.csv")
        dataset = full_dataset.sample(frac=0.2)
        training_dataset = dataset.sample(frac=0.75)
        testing_dataset = dataset.drop(training_dataset.index)
    elif dataset_name in ["CICIDS2017"]:
        dataset = pd.read_csv("/home/haochu/Documents/PoisoningAttack/Dataset/CICIDS2017/CICIDS2017_full.csv")
        training_dataset = dataset.sample(frac=0.75)
        testing_dataset = dataset.drop(training_dataset.index)
    elif dataset_name in ["N-BaIoT"]:
        dataset = pd.read_csv("/home/haochu/Documents/PoisoningAttack/Dataset/N-BaIoT/Full/N-baiot5.csv")
        training_dataset = dataset.sample(frac=0.70)
        testing_dataset = dataset.drop(training_dataset.index)

    training_inputs, training_labels = preprocessing_training_dataset(training_dataset, dataset_name)
    testing_inputs, testing_labels = preprocessing_training_dataset(testing_dataset, dataset_name)

    # split dataset according to iid flag
    if iid:
        # shuffle data
        shuffled_indices = torch.randperm(len(training_inputs))
        training_inputs = training_inputs[shuffled_indices]
        training_labels = torch.Tensor(training_labels.to_numpy())[shuffled_indices]

        # partition data into num_clients
        split_size = len(training_dataset) // num_clients
        split_datasets = list(
            zip(
                torch.split(torch.Tensor(training_inputs), split_size),
                torch.split(torch.Tensor(training_labels), split_size)
            )
        )
        if attack_mode in ['Label-Flipping']:
            datasets = list()
            # Label Flipping for 4 attackers
            for i in range(num_clients):
                if 0 < i < 5:
                    labels = split_datasets[i][1].cpu().detach().numpy()
                    new_labels = np.array([abs(s-1) for s in labels])
                    datasets.append((split_datasets[i][0], torch.Tensor(new_labels)))
                else:
                    datasets.append((split_datasets[i][0], split_datasets[i][1]))
            # finalize bunches of local datasets
            local_datasets = [
                CustomTensorDataset(local_dataset)
                for local_dataset in datasets
                ]
        else:
            # finalize bunches of local datasets
            local_datasets = [
                CustomTensorDataset(local_dataset)
                for local_dataset in split_datasets
                ]
        
        shuffled_indices = torch.randperm(len(testing_inputs))
        testing_inputs = testing_inputs[shuffled_indices]
        testing_labels = torch.Tensor(testing_labels.to_numpy())[shuffled_indices]

        testing_dataset = list(
            zip(
                torch.split(torch.Tensor(testing_inputs), len(testing_dataset)), 
                torch.split(torch.Tensor(testing_labels), len(testing_dataset))
                )
            )

    return local_datasets, CustomTensorDataset(testing_dataset[0])

# def _create_datasets(data_path, dataset_name, num_clients, num_shards, iid):
#     """Split the whole dataset in IID or non-IID manner for distributing to clients."""
#     dataset_name = dataset_name.upper()
#     # get dataset from torchvision.datasets if exists
#     if hasattr(torchvision.datasets, dataset_name):
#         # set transformation differently per dataset
#         if dataset_name in ["CIFAR10"]:
#             transform = torchvision.transforms.Compose(
#                 [
#                     torchvision.transforms.ToTensor(),
#                     torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#                 ]
#             )
#         elif dataset_name in ["MNIST"]:
#             transform = torchvision.transforms.ToTensor()
        
#         # prepare raw training & test datasets
#         training_dataset = torchvision.datasets.__dict__[dataset_name](
#             root=data_path,
#             train=True,
#             download=True,
#             transform=transform
#         )
#         test_dataset = torchvision.datasets.__dict__[dataset_name](
#             root=data_path,
#             train=False,
#             download=True,
#             transform=transform
#         )
#     else:
#         # dataset not found exception
#         error_message = f"...dataset \"{dataset_name}\" is not supported or cannot be found in TorchVision Datasets!"
#         raise AttributeError(error_message)

#     # unsqueeze channel dimension for grayscale image datasets
#     if training_dataset.data.ndim == 3: # convert to NxHxW -> NxHxWx1
#         training_dataset.data.unsqueeze_(3)
#     num_categories = np.unique(training_dataset.targets).shape[0]
    
#     if "ndarray" not in str(type(training_dataset.data)):
#         training_dataset.data = np.asarray(training_dataset.data)
#     if "list" not in str(type(training_dataset.targets)):
#         training_dataset.targets = training_dataset.targets.tolist()
    
#     # split dataset according to iid flag
#     if iid:
#         # shuffle data
#         shuffled_indices = torch.randperm(len(training_dataset))
#         training_inputs = training_dataset.data[shuffled_indices]
#         training_labels = torch.Tensor(training_dataset.targets)[shuffled_indices]

#         # partition data into num_clients
#         split_size = len(training_dataset) // num_clients
#         split_datasets = list(
#             zip(
#                 torch.split(torch.Tensor(training_inputs), split_size),
#                 torch.split(torch.Tensor(training_labels), split_size)
#             )
#         )

#         # finalize bunches of local datasets
#         local_datasets = [
#             CustomTensorDataset(local_dataset, transform=transform)
#             for local_dataset in split_datasets
#             ]
#     else:
#         # sort data by labels
#         sorted_indices = torch.argsort(torch.Tensor(training_dataset.targets))
#         training_inputs = training_dataset.data[sorted_indices]
#         training_labels = torch.Tensor(training_dataset.targets)[sorted_indices]

#         # partition data into shards first
#         shard_size = len(training_dataset) // num_shards #300
#         shard_inputs = list(torch.split(torch.Tensor(training_inputs), shard_size))
#         shard_labels = list(torch.split(torch.Tensor(training_labels), shard_size))

#         # sort the list to conveniently assign samples to each clients from at least two classes
#         shard_inputs_sorted, shard_labels_sorted = [], []
#         for i in range(num_shards // num_categories):
#             for j in range(0, ((num_shards // num_categories) * num_categories), (num_shards // num_categories)):
#                 shard_inputs_sorted.append(shard_inputs[i + j])
#                 shard_labels_sorted.append(shard_labels[i + j])
                
#         # finalize local datasets by assigning shards to each client
#         shards_per_clients = num_shards // num_clients
#         local_datasets = [
#             CustomTensorDataset(
#                 (
#                     torch.cat(shard_inputs_sorted[i:i + shards_per_clients]),
#                     torch.cat(shard_labels_sorted[i:i + shards_per_clients]).long()
#                 ),
#                 transform=transform
#             ) 
#             for i in range(0, len(shard_inputs_sorted), shards_per_clients)
#         ]
#     return local_datasets, test_dataset
