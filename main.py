# import
from src.evaluate import kFold_validation
from src.project_parameters import ProjectPrameters
from src.train import train
from src.predict import predict

# def


def main(projectParams):
    result = None
    if projectParams.mode == 'train':
        result = train(projectParams=projectParams)
    elif projectParams.mode == 'evaluate':
        kFold_validation(projectParams=projectParams)
        result = 1
    elif projectParams.mode == 'predict':
        result = predict(projectParams=projectParams)
        print(('{},'*projectParams.numClasses).format(*
                                                      projectParams.dataType.keys())[:-1])
        print(result)
    return result


if __name__ == '__main__':
    # project parameters
    projectParams = ProjectPrameters().parse()

    # main
    result = main(projectParams=projectParams)
