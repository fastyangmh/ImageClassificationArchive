# import
import torch

# def


def load_checkpoint(model, checkpointPath):
    checkpoint = torch.load(checkpointPath)
    model.load_state_dict(checkpoint['state_dict'])
    return model
