# import
from src.project_parameters import ProjectPrameters
from ray import tune, init
from ray.tune.schedulers import ASHAScheduler
from ray.tune import CLIReporter
from src.train import train
from functools import partial
from copy import copy
import numpy as np

# def


def get_hyperparameter_space(projectParams):
    hparamsSpace = {
        # the random integer between low and high
        'trainIter': tune.randint(lower=10, upper=200),
        'lrSchedulerStepSize': tune.randint(lower=1, upper=10),
        # the random floating-point number between low and high
        'lr': tune.uniform(lower=1e-4, upper=1e-1),
        # the stochastic choices the one element from the given list
        'optimizer': tune.choice(['adam', 'sgd']),
        'lrScheduler': tune.choice(['cosine', 'step'])
    }
    if projectParams.useEarlyStopping:
        # the random integer between low and high
        hparamsSpace['earlyStoppingPatience'] = tune.randint(lower=1, upper=10)
    return hparamsSpace


def parse_result(result):
    lossDict = {}
    accuracyDict = {}
    for stage in ['train', 'val', 'test']:
        loss, accuracy = result[stage][0].values()
        lossDict[stage] = loss
        accuracyDict[stage] = accuracy
    return lossDict, accuracyDict


def set_projectParams(hparamsSpace, projectParams):
    for key, value in hparamsSpace.items():
        if type(value) == str:
            exec('projectParams.{}="{}"'.format(key, value))
        else:
            exec('projectParams.{}={}'.format(key, value))
        if key == 'trainIter' and not projectParams.useEarlyStopping:
            exec('projectParams.valIter={}'.format(value))
    return projectParams


def tuning_function(hparamsSpace, projectParams):
    if projectParams.tuneDebug:
        sumOfHparams = sum(
            [value for value in hparamsSpace.values() if type(value) is not str])
        tune.report(diffAccuracy=sumOfHparams)
    else:
        # use the copy API to prevent the object was modified
        projectParams = set_projectParams(
            hparamsSpace=hparamsSpace, projectParams=copy(projectParams))
        result = train(projectParams=projectParams)
        lossDict, accuracyDict = parse_result(result=result)
        diffAccuracy = sum([1-value for value in accuracyDict.values()])
        tune.report(trainLoss=lossDict['train'],
                    valLoss=lossDict['val'],
                    testLoss=lossDict['test'],
                    trainAccuracy=accuracyDict['train'],
                    valAccuracy=accuracyDict['val'],
                    testAccuracy=accuracyDict['test'],
                    diffAccuracy=diffAccuracy)


def tuning(projectParams):
    # set the mode value as train in the projectParams
    projectParams.mode = 'train'

    # set hyperparameter space
    hparamsSpace = get_hyperparameter_space(projectParams=projectParams)

    # set tune scheduler
    scheduler = ASHAScheduler(metric='diffAccuracy', mode='min')

    # set tune reporter
    reporter = CLIReporter(metric_columns=[
                           'trainLoss', 'valLoss', 'testLoss', 'trainAccuracy', 'valAccuracy', 'testAccuracy',  'diffAccuracy'])

    # run the hyperparameter search
    init(dashboard_host='0.0.0.0')
    tuningResult = tune.run(run_or_experiment=partial(tuning_function, projectParams=projectParams),
                            resources_per_trial={
                                'cpu': projectParams.tuneCPU, 'gpu': projectParams.tuneGPU},
                            config=hparamsSpace,
                            num_samples=projectParams.tuneIter,
                            scheduler=scheduler,
                            local_dir='./save/tuning_logs',
                            progress_reporter=reporter)

    # get the best trial
    bestTrial = tuningResult.get_best_trial('diffAccuracy', 'min', 'last')

    # run the train with the best parameters from the best trial
    if not projectParams.tuneDebug:
        projectParams = set_projectParams(
            hparamsSpace=bestTrial.config, projectParams=projectParams)
        result = train(projectParams=projectParams)
        result['tune'] = tuningResult
    else:
        result = {'tune': tuningResult}
    print('best trial name: {}'.format(bestTrial))
    print('best trial result: {}'.format(
        bestTrial.last_result['diffAccuracy']))
    print('best trial config: {}'.format(bestTrial.config))
    print('best trial config command: --numWorkers {}{}'.format(projectParams.numWorkers,
                                                                (' --{} {}'*len(bestTrial.config)).format(*np.concatenate(list(zip(bestTrial.config.keys(), bestTrial.config.values()))))))
    return result


if __name__ == '__main__':
    # project parameters
    projectParams = ProjectPrameters().parse()

    # tune
    result = tuning(projectParams=projectParams)
