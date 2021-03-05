# import
from src.project_parameters import ProjectPrameters
from glob import glob
from os.path import join
import numpy as np
from os import makedirs
from shutil import copy2, copytree, rmtree
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
from src.train import train

# def


def dataPath2Dataset(projectParams):
    data, label = [], []
    for stage in ['train', 'val']:
        for dType in projectParams.dataType.keys():
            temp = sorted(
                glob(join(projectParams.dataPath, '{}/{}/*.png'.format(stage, dType))))
            data += temp
            label += [projectParams.dataType[dType]]*len(temp)
    return {'data': np.array(data), 'label': np.array(label)}


def copyFile2Dst(files, dst, projectParams):
    for dType in projectParams.dataType.keys():
        makedirs(join(dst, dType), exist_ok=True)
        for file in files:
            if dType in file:
                copy2(src=file, dst=join(dst, dType))


def get_kFold_dataset(projectParams, dataset):
    sKF = StratifiedKFold(n_splits=projectParams.kFoldValue, shuffle=True)
    for idx, (trainIndex, valIndex) in tqdm(enumerate(sKF.split(X=dataset['data'], y=dataset['label'])), total=projectParams.kFoldValue):
        xTrain = dataset['data'][trainIndex]
        xVal = dataset['data'][valIndex]
        dst = join(projectParams.kFoldDataPath, 'kFold{}'.format(idx+1))
        makedirs(join(dst, 'train'), exist_ok=True)
        makedirs(join(dst, 'val'), exist_ok=True)
        copyFile2Dst(files=xTrain, dst=join(dst, 'train'),
                     projectParams=projectParams)
        copyFile2Dst(files=xVal, dst=join(dst, 'val'),
                     projectParams=projectParams)
        copytree(src=join(projectParams.dataPath, 'test'),
                 dst=join(dst, 'test'))


def get_kFold_result(projectParams):
    print('start k-fold validation')
    results = {}
    directories = sorted(glob(join(projectParams.kFoldDataPath, '*/')))
    for idx, directory in enumerate(directories):
        print('-'*30)
        print('\nk-fold validation: {}/{}'.format(idx+1, projectParams.kFoldValue))
        projectParams.dataPath = directory
        results[idx+1] = train(projectParams=projectParams)
    return results


def parse_kFold_result(results):
    trainLoss, valLoss, testLoss = [], [], []
    accTrain, accVal, accTest = [], [], []
    for v in results.values():
        eachStageResult = {stage: list(v[stage][0].values())
                           for stage in ['train', 'val', 'test']}
        trainLoss.append(eachStageResult['train'][0])
        accTrain.append(eachStageResult['train'][1])
        valLoss.append(eachStageResult['val'][0])
        accVal.append(eachStageResult['val'][1])
        testLoss.append(eachStageResult['test'][0])
        accTest.append(eachStageResult['test'][1])
    return {'train': (trainLoss, accTrain), 'val': (valLoss, accVal), 'test': (testLoss, accTest)}


def kFold_validation(projectParams):
    # load files to dataset
    dataset = dataPath2Dataset(projectParams=projectParams)

    # get split index and copy file to destination
    get_kFold_dataset(projectParams=projectParams, dataset=dataset)

    # train the model by the kFoldDataPath
    results = get_kFold_result(projectParams=projectParams)

    # parse the results
    results = parse_kFold_result(results=results)

    # display information
    print('-'*20)
    print('\nthe result of the model with given parameters is as follows.')
    print('k-Fold validation training loss mean:\t{} ± {}'.format(np.mean(
        results['train'][0]), (max(results['train'][0])-min(results['train'][0]))/2))
    print('Train accuracy mean:\t{} ± {}'.format(np.mean(
        results['train'][1]), (max(results['train'][1])-min(results['train'][1]))/2))
    print('Validation accuracy mean:\t{} ± {}'.format(
        np.mean(results['val'][1]), (max(results['val'][1])-min(results['val'][1]))/2))
    print('Test accuracy mean:\t{} ± {}'.format(np.mean(
        results['test'][1]), (max(results['test'][1])-min(results['test'][1]))/2))

    # remove kFoldDataPath
    rmtree(path=projectParams.kFoldDataPath)


if __name__ == '__main__':
    # project parameters
    projectParams = ProjectPrameters().parse()

    # k-fold validation
    if projectParams.predefinedTask is None:
        kFold_validation(projectParams=projectParams)
