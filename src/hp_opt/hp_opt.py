from random import random
import optuna
import pytorch_lightning as pl
from calibrator.cal_trainer import *
from calibrator.local_net import *
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import EarlyStopping
from calibrator.replicator import *
from algorithms.networks.networks import *
from algorithms.trainers.trainers import *
from optuna.integration import PyTorchLightningPruningCallback


class OptunaPruningCallback(PyTorchLightningPruningCallback, pl.Callback):
    pass

def objective(trial, kwargs, train_loader, val_loader, wandb_logger):
    cuda_device = kwargs.cuda_device
    total_epochs = kwargs.optuna_epochs
    
    csv_logger = CSVLogger(
        save_dir="optuna_logs",
        name=f"trial_{trial.number}"
    )

    # Suggest hyperparameters
    # lambda_kl = trial.suggest_float("lambda_kl", 1.0, 10.0)
    # alpha1 = trial.suggest_float("alpha1", 1.0, 10.0)
    # log_var_initializer = trial.suggest_float("log_var_initializer", 0.001, 10.0)
    alpha_sim = trial.suggest_float("alpha_sim", 0.05, 1.0)
    alpha_cls = trial.suggest_float("alpha_cls", 0.05, 1.0)
    #smoothing = trial.suggest_float("smoothing", 0.001, 0.2)
    
    # kwargs.models.lambda_kl = lambda_kl
    # kwargs.models.alpha1 = alpha1
    # kwargs.models.log_var_initializer = log_var_initializer
    kwargs.models.alpha_sim = alpha_sim
    kwargs.models.alpha_cls = alpha_cls
    #kwargs.models.smoothing = smoothing

    # Build your model with these hyperparameters
    model = AuxTrainerV2(kwargs.models, num_classes=kwargs.dataset.num_classes, feature_dim=kwargs.dataset.feature_dim, similarity_dim=kwargs.similarity_dim) #AuxTrainer(kwargs.models, num_classes=kwargs.dataset.num_classes)    

    trainer = pl.Trainer(
            max_epochs=total_epochs,
            accelerator="cuda",
            devices=[cuda_device],
            logger=csv_logger,
            enable_progress_bar=False,
            enable_model_summary=False,
            deterministic=True,
            callback=[EarlyStopping(monitor="val_total", #val_kl
                                      patience=5, #5
                                      mode="min", 
                                      verbose=True, 
                                      min_delta=0.0)]
    )
    
    # Train and validate
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # Use final validation loss as objective
    optuna_loss = trainer.callback_metrics["optuna_loss"].item()
    val_kl = trainer.callback_metrics["val_kl"].item()
    val_con_loss = trainer.callback_metrics["val_con_loss"].item()
    if kwargs.multi_obj:
        return val_kl, val_con_loss
    else:    
        return val_kl #optuna_loss
    
    
def replicator_objective(trial, kwargs, train_loader, val_loader, test_loader, wandb_logger):
    cuda_device = kwargs.cuda_device
    total_epochs = kwargs.optuna_epochs   
    
    # Reproducibility
    seed = kwargs.optuna_seed #+ trial.number  # Different seed for each trial
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 

    # Suggest hyperparameters        
    lin_comb = trial.suggest_float("lin_comb", 0.6, 0.99, log=True) # 0.01 # trial.suggest_categorical("lin_comb", [0.85, 0.9, 0.95, 0.97, 0.99]) # 0.7, 0.8, 
    ceiling = trial.suggest_float("ceiling", 1e-3, 3e-1, log=True) # ceiling = trial.suggest_categorical("ceiling", [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5])
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True) # lr = trial.suggest_categorical("lr", [0.001, 0.005, 0.01, 0.0005])
    n_steps = trial.suggest_categorical("n_steps", [10, 25, 50, 75, 100]) # 20, 30, 70, 85
    eta = trial.suggest_float("step_size", 1e-4, 1e-1, log=True) # eta = trial.suggest_categorical("step_size", [0.01, 0.05, 0.1, 0.005, 0.001]) # initial step size vaues for the potential replicator
    optimizer = trial.suggest_categorical("optimizer", ['adam', 'adamw']) # 'sgd'
    hidden_dim = trial.suggest_categorical("feature_dim", [8, 16, 32, 64, 128, 256, 512]) # 10
    
    # kwargs.models.lin_comb = lin_comb
    # kwargs.models.ceiling = ceiling
    # kwargs.models.feature_dim = hidden_dim

    # Build your model with these hyperparameters
    calibrator = PotentialReplicatorCalibrator(                      
        n_classes =             kwargs.dataset.num_classes,
        data =                  kwargs.data,
        n_steps =               n_steps, # n_steps,
        hidden =                hidden_dim,
        lin_comb =              lin_comb, # lin_comb,
        ceiling =               ceiling,  
        optimizer =             optimizer, # kwargs.models.
        lr =                    lr,
        weight_decay=           kwargs.models.weight_decay,
        epochs =                total_epochs,        
        eta =                   eta,       
        eps =                   1e-8,        
        kl_reg =                kwargs.models.kl_reg, 
        l2_reg =                kwargs.models.l2_reg,
        fit_stage =             kwargs.models.fit_stage,
        potential =             kwargs.models.potential
        )   
    
    # Train and validate
    calibrator.fit(train_loader, val_loader, test_loader, optuna_=True, trial=trial, device=cuda_device)

    # Use final validation loss as objective
    optuna_loss = calibrator.best_val_nll 
    
    return optuna_loss    


def pretrain_objective(trial, kwargs, train_loader, val_loader, wandb_logger):
    cuda_device = kwargs.cuda_device
    total_epochs = kwargs.optuna_epochs   
    monitor_metric = 'val_acc' # 'val_loss'
    
    # Reproducibility
    seed = kwargs.optuna_seed #+ trial.number  # Different seed for each trial
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 

    # Suggest hyperparameters        
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True) # lr = trial.suggest_categorical("lr", [0.001, 0.005, 0.01, 0.0005])        
    optimizer = trial.suggest_categorical("name", ['adam', 'adamw']) # 'sgd'4
    wd = trial.suggest_float("weight_decay", 1e-4, 1e-1, log=True)    
    
    kwargs.models.optimizer.name = optimizer
    kwargs.models.optimizer.lr = lr    
    kwargs.models.optimizer.weight_decay = wd
    
    if kwargs.data == 'cifar10':
        pl_model = Cifar10Model(kwargs.models)   
    elif kwargs.data == 'cifar100':
        pl_model = Cifar100Model(kwargs.models)   
    elif kwargs.data == 'tissue':
        pl_model = MedMnistModel(kwargs.models)   
        
    trainer = pl.Trainer(
            max_epochs=total_epochs,
            accelerator="cuda",
            devices=[cuda_device],
            logger=wandb_logger,
            check_val_every_n_epoch=1,            
            deterministic=True,
            callbacks=[ ClearCacheCallback(),
                       OptunaPruningCallback(trial, monitor=monitor_metric)
                       ])
        
    trainer.fit(pl_model, train_loader, val_loader)

    # Use final validation loss as objective
    optuna_loss = trainer.callback_metrics[monitor_metric].item() # trainer.callback_metrics["val_loss"].item() 
    
    return optuna_loss    



def print_callback(study, trial):
    print(f"Trial {trial.number} finished with value: {trial.value}")
    print(f"  Params: {trial.params}\n")


def multi_obj_print_callback(study, trial):
    print(f"Trial {trial.number} finished with values: {trial.values}")
    print(f"  Params: {trial.params}\n")
    
    