# import
import torch
from glob import glob
from os.path import join

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
    for dType in projectParams.dataType.keys():
        dataWeight[dType] = len(
            glob(join(projectParams.dataPath, 'train/{}/*.png'.format(dType))))
    projectParams.dataWeight = {
        dType: 1-(dataWeight[dType]/sum(dataWeight.values())) for dType in dataWeight.keys()}
    return projectParams
