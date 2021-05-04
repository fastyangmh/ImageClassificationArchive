# import
import torch
from glob import glob
from os.path import join
from ruamel.yaml import safe_load
from torchvision import transforms

# def


def load_checkpoint(model, use_cuda, checkpoint_path):
    map_location = torch.device(
        device='cuda') if use_cuda else torch.device(device='cpu')
    checkpoint = torch.load(f=checkpoint_path, map_location=map_location)
    if model.loss_function.weight is None:
        # delete the loss_function.weight in the checkpoint, because this key does not work while loading the model.
        del checkpoint['state_dict']['loss_function.weight']
    else:
        # assign the new loss_function weight to the checkpoint
        checkpoint['state_dict']['loss_function.weight'] = model.loss_function.weight
    model.load_state_dict(checkpoint['state_dict'])
    return model


def get_files(file_path, file_type):
    files = []
    if type(file_type) != list:
        file_type = [file_type]
    for v in file_type:
        files += sorted(glob(join(file_path, '*.{}'.format(v))))
    return files


def calculate_data_weight(classes, data_path):
    data_weight = {}
    for c in classes.keys():
        files = get_files(file_path=join(
            data_path, 'train/{}'.format(c)), file_type=['jpg', 'png'])
        data_weight[c] = len(files)
    data_weight = {c: 1-(data_weight[c]/sum(data_weight.values()))
                   for c in data_weight.keys()}
    return data_weight


def load_yaml(file_path):
    with open(file_path, 'r') as f:
        config = safe_load(f)
    return config


def get_transform_from_file(file_path):
    transform_dict = {}
    transform_config = load_yaml(file_path=file_path)
    for stage in transform_config.keys():
        transform_dict[stage] = []
        for name, value in transform_config[stage].items():
            if value is None:
                transform_dict[stage].append(
                    eval('transforms.{}()'.format(name)))
            else:
                if type(value) is dict:
                    value = ('{},'*len(value)).format(*
                                                      ['{}={}'.format(a, b) for a, b in value.items()])
                transform_dict[stage].append(
                    eval('transforms.{}({})'.format(name, value)))
        transform_dict[stage] = transforms.Compose(transform_dict[stage])
    return transform_dict
