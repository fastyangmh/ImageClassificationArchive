# import
import torch
from glob import glob
from os.path import isfile, join
from ruamel.yaml import safe_load
from torchvision import transforms

# def


def load_checkpoint(model, num_classes, use_cuda, checkpoint_path):
    """Load a checkpoint for the neural network model.

    Args:
        model (timm.models or nn.Module): the timm model or the self-defined backbone model.
        num_classes (int): the number of classes.
        use_cuda (bool): whether to use Cuda to train the model. if True which will train the model on CPU. if False which will train on GPU.
        checkpoint_path (str): the path of the pre-trained model checkpoint.

    Returns:
        timm.models or nn.Module: the timm model or the self-defined backbone model.
    """
    map_location = torch.device(
        device='cuda') if use_cuda else torch.device(device='cpu')
    checkpoint = torch.load(f=checkpoint_path, map_location=map_location)
    for k in checkpoint['state_dict'].keys():
        if 'classifier.bias' in k or 'classifier.weight' in k:
            if checkpoint['state_dict'][k].shape[0] != num_classes:
                temp = checkpoint['state_dict'][k]
                checkpoint['state_dict'][k] = torch.stack(
                    [temp.mean(0)]*num_classes, 0)
    if model.loss_function.weight is None:
        # delete the loss_function.weight in the checkpoint, because this key does not work while loading the model.
        if 'loss_function.weight' in checkpoint['state_dict']:
            del checkpoint['state_dict']['loss_function.weight']
    else:
        # assign the new loss_function weight to the checkpoint
        checkpoint['state_dict']['loss_function.weight'] = model.loss_function.weight
    model.load_state_dict(checkpoint['state_dict'])
    return model


def get_files(filepath, file_type):
    """Return a sorted list of files in the given path.

    Args:
        filepath (str): the file path of the backbone model.
        file_type (list): a list of filename extension.

    Returns:
        list: a sorted list of files.
    """
    files = []
    if type(file_type) != list:
        file_type = [file_type]
    for v in file_type:
        files += sorted(glob(join(filepath, '*.{}'.format(v))))
    return files


def calculate_data_weight(classes, data_path):
    """Calculate the weight of a set of classes .

    Args:
        classes (list): the classes of data. if use a predefined dataset, please set value as None.
        data_path (str): the data path.

    Returns:
        dict: the dictionary contains the weight of each class. it is used for the weight of loss function.
    """
    data_weight = {}
    for c in classes:
        files = get_files(filepath=join(
            data_path, 'train/{}'.format(c)), file_type=['jpg', 'png'])
        data_weight[c] = len(files)
    data_weight = {c: 1-(data_weight[c]/sum(data_weight.values()))
                   for c in data_weight.keys()}
    return data_weight


def load_yaml(filepath):
    """Load a yaml config file .

    Args:
        filepath (str): the file path of the backbone model.

    Returns:
        dict: the dictionary contains the content of yaml.
    """
    with open(filepath, 'r') as f:
        config = safe_load(f)
    return config


def get_transform_from_file(filepath):
    """Load transform from file path for image transformations.

    Args:
        filepath (str): the file path of the backbone model.

    Returns:
        dict: the dictionary contains the transforms of train, validation, test, and predict.
    """
    if filepath is None:
        return {}.fromkeys(['train', 'val', 'test', 'predict'], None)
    elif isfile(filepath):
        transform_dict = {}
        transform_config = load_yaml(filepath=filepath)
        for stage in transform_config.keys():
            transform_dict[stage] = []
            if type(transform_config[stage]) != dict:
                transform_dict[stage] = None
                continue
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
    else:
        assert False, 'please check the transform config path: {}'.format(
            filepath)
