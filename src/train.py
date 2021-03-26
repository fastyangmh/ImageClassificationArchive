# import
from src.project_parameters import ProjectPrameters
import pytorch_lightning as pl
from src.data_preparation import MyDataModule
from src.model import create_model
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from src.utils import calculate_data_weight
import warnings
warnings.filterwarnings("ignore")

# def


def get_trainer(projectParams):
    callbacks = [ModelCheckpoint(monitor='validation epoch accuracy'),
                 LearningRateMonitor(logging_interval='epoch')]
    if projectParams.useEarlyStopping:
        callbacks.append(EarlyStopping(monitor='validation epoch loss',
                                       patience=projectParams.earlyStoppingPatience, mode='min'))
    return pl.Trainer(gpus=projectParams.gpus,
                      max_epochs=projectParams.trainIter,
                      check_val_every_n_epoch=projectParams.valIter,
                      default_root_dir=projectParams.savePath,
                      callbacks=callbacks,
                      log_gpu_memory='all',
                      num_sanity_val_steps=0,
                      profiler=projectParams.report,
                      deterministic=True,
                      weights_summary=projectParams.weightsSummary)


def train(projectParams):
    pl.seed_everything(seed=projectParams.randomSeed)
    if projectParams.useBalance and projectParams.predefinedTask is None:
        projectParams = calculate_data_weight(projectParams=projectParams)
    dataset = MyDataModule(projectParams=projectParams)
    model = create_model(projectParams=projectParams)
    trainer = get_trainer(projectParams=projectParams)
    trainer.fit(model=model, datamodule=dataset)
    result = {'trainer': trainer,
              'model': model}
    trainer.callback_connector.configure_progress_bar().disable()
    for stage, dataLoader in zip(['train', 'val', 'test'], [dataset.train_dataloader(), dataset.val_dataloader(), dataset.test_dataloader()]):
        print('\ntest the {} dataset'.format(stage))
        print('the {} dataset confusion matrix:'.format(stage))
        result[stage] = trainer.test(test_dataloaders=dataLoader)
    trainer.callback_connector.configure_progress_bar().enable()
    return result


if __name__ == '__main__':
    # project parameters
    projectParams = ProjectPrameters().parse()

    # train the model
    result = train(projectParams=projectParams)
