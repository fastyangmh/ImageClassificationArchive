# import
from src.project_parameters import ProjectPrameters
import pytorch_lightning as pl
from src.data_preparation import MyDataModule
from src.model import Net
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import warnings
warnings.filterwarnings("ignore")

# def


def get_trainer(projectParams):
    callbacks = [ModelCheckpoint(monitor='validation epoch accuracy'),
                 LearningRateMonitor(logging_interval='epoch')]
    return pl.Trainer(gpus=projectParams.gpus,
                      max_epochs=projectParams.trainIter,
                      amp_backend='native',
                      check_val_every_n_epoch=projectParams.valIter,
                      default_root_dir=projectParams.savePath,
                      callbacks=callbacks,
                      log_gpu_memory='all',
                      num_sanity_val_steps=0,
                      profiler=projectParams.report,
                      weights_summary=projectParams.weightsSummary)


def train(projectParams):
    result = {}
    pl.seed_everything(seed=projectParams.randomSeed)
    myDataModule = MyDataModule(projectParams=projectParams)
    model = Net(projectParams=projectParams)
    trainer = get_trainer(projectParams=projectParams)
    trainer.fit(model=model, datamodule=myDataModule)
    result = {'trainer': trainer,
              'model': model,
              'train': trainer.test(test_dataloaders=myDataModule.train_dataloader()),
              'val': trainer.test(test_dataloaders=myDataModule.val_dataloader()),
              'test': trainer.test(test_dataloaders=myDataModule.test_dataloader())}
    return result


if __name__ == '__main__':
    # project parameters
    projectParams = ProjectPrameters().parse()

    # train the model
    result = train(projectParams=projectParams)
