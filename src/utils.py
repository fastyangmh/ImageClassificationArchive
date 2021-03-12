# import
import torch
from glob import glob
from os.path import join

# def


def load_checkpoint(model, checkpointPath):
    checkpoint = torch.load(checkpointPath)
    if 'criterion.weight' in checkpoint['state_dict']:
        # delete the critreion.weight in the checkpoint, because this key does not work while loading the model.
        del checkpoint['state_dict']['criterion.weight']
    model.load_state_dict(checkpoint['state_dict'])
    return model


def calculate_data_weight(projectParams):
    dataWeight = {}
    for dType in projectParams.dataType.keys():
        dataWeight[dType] = len(
            glob(join(projectParams.dataPath, 'train/{}/*.png'.format(dType))))
    projectParams.dataWeight = {
        dType: 1-(dataWeight[dType]/sum(dataWeight.values())) for dType in dataWeight.keys()}
    return projectParams
