# import
import torch
from glob import glob
from os.path import join
from ruamel.yaml import safe_load

# def


def load_checkpoint(model, projectParams):
    checkpoint = torch.load(projectParams.checkpointPath, map_location=torch.device(
        'cuda' if projectParams.useCuda else 'cpu'))
    if model.criterion.weight is None:
        # delete the critreion.weight in the checkpoint, because this key does not work while loading the model.
        del checkpoint['state_dict']['criterion.weight']
    else:
        # assign the new criterion weight to the checkpoint
        checkpoint['state_dict']['criterion.weight'] = model.criterion.weight
    model.load_state_dict(checkpoint['state_dict'])
    return model


def calculate_data_weight(projectParams):
    dataWeight = {}
    for c in projectParams.classes.keys():
        files = []
        for ext in ['png', 'jpg']:
            files += glob(join(projectParams.dataPath,
                               'train/{}/*.{}'.format(c, ext)))
        dataWeight[c] = len(files)
    projectParams.dataWeight = {
        c: 1-(dataWeight[c]/sum(dataWeight.values())) for c in dataWeight.keys()}
    return projectParams


def load_yaml(filePath):
    with open(filePath, 'r') as f:
        config = safe_load(f)
    return config
