# import
import torch

# def


def load_checkpoint(model, checkpointPath):
    checkpoint = torch.load(checkpointPath)
    if 'criterion.weight' in checkpoint['state_dict']:
        # delete the critreion.weight in the checkpoint, because this key does not work while loading the model.
        del checkpoint['state_dict']['criterion.weight']
    model.load_state_dict(checkpoint['state_dict'])
    return model
