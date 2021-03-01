# import
from src.project_parameters import ProjectPrameters
from src.train import train

# def


def main(projectParams):
    if projectParams.mode == 'train':
        result = train(projectParams=projectParams)
    return result


if __name__ == '__main__':
    # project parameters
    projectParams = ProjectPrameters().parse()

    # main
    result = main(projectParams=projectParams)
