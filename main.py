# import
from src.evaluate import kFold_validation
from src.project_parameters import ProjectPrameters
from src.train import train
from src.predict import Predict
from src.tune import tuning

# def


def main(projectParams):
    result = None
    if projectParams.mode == 'train':
        result = train(projectParams=projectParams)
    elif projectParams.mode == 'evaluate':
        if projectParams.predefinedTask is None:
            kFold_validation(projectParams=projectParams)
        else:
            print('Temporarily does not support predefined tasks.')
    elif projectParams.mode == 'predict':
        result = Predict(projectParams=projectParams).get_result(
            dataPath=projectParams.dataPath)
        print(('{},'*projectParams.numClasses).format(*
                                                      projectParams.dataType.keys())[:-1])
        print(result)
    elif projectParams.mode == 'tune':
        result = tuning(projectParams=projectParams)
    return result


if __name__ == '__main__':
    # project parameters
    projectParams = ProjectPrameters().parse()

    # main
    result = main(projectParams=projectParams)
