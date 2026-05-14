from matplotlib.pylab import rint
import pytorch_lightning as pl
import hydra
from hydra import initialize, compose
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, open_dict, OmegaConf
from src.actions.pretrain import *
from src.actions.test import *
from src.utils.utils import *
from src.actions.calibrate import *
from src.actions.quantize import *
from src.actions.replicate import *
from src.actions.competition import *
from src.actions.viz_and_test import *
from pytorch_lightning.loggers import WandbLogger
import time
from datetime import datetime
import os
import sys
import wandb
    
    
def main(cfg: DictConfig):
    kwargs = cfg #OmegaConf.create(cfg)  
    
    now = datetime.now()
    start = time.time()    
        
    dataset_name = kwargs.data
    model_name = kwargs.models_map[dataset_name]
    epochs = kwargs.checkpoint.epochs
    if epochs == 9:
        model_class = 'resnet'
        kwargs.dataset.feature_dim = 2048
    elif kwargs.checkpoint.epochs == 5:
        model_class = 'vit'
        kwargs.dataset.feature_dim = 768
    elif kwargs.checkpoint.epochs == 20:
        model_class = 'convnext'
        kwargs.dataset.feature_dim = 768
    else:
        model_class = 'ftt'
        kwargs.dataset.feature_dim = 2048
        if not kwargs.data == 'weather':
            raise ValueError(
                f'Checkpoint not corresponding to a trained modl! {kwargs.checkpoint.epochs} was given but only 9 and 20 are supported')

    base_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
    base_dir = os.path.join(os.path.dirname(os.path.dirname(base_dir)), 'result')
    
    exp_name = f'{kwargs.exp_name}_{kwargs.data}_{now.strftime("%m%d_%H%M")}' #target
    if kwargs.use_optuna:
        exp_name = 'optuna_'+ exp_name #{kwargs.data}_{now.strftime("%m%d_%H%M")}' 
    if kwargs.use_wandb:
        if kwargs.resume_training and kwargs.wandb_id:
             wandb_logger = WandbLogger(name=exp_name, project=kwargs.wandb_project, entity=kwargs.wandb_entity, save_dir=base_dir, offline=kwargs.offline, id=kwargs.wandb_id, resume='allow')
        else:
            wandb_logger = WandbLogger(name=exp_name, project=kwargs.wandb_project, entity=kwargs.wandb_entity, save_dir=base_dir, offline=kwargs.offline)        
    else:
        wandb_logger = WandbLogger(name=exp_name, project='Test', entity=kwargs.wandb_entity, save_dir=base_dir, offline=kwargs.offline)
    #if kwargs.use_optuna:
    #    wandb_optuna_logger = WandbLogger(name=optuna_exp_name, project=kwargs.wandb_project, entity=kwargs.wandb_entity, save_dir=base_dir, offline=kwargs.offline)
    #else: 
    #    wandb_optuna_logger = None
    kwargs.wandb_id = wandb_logger.version

        
    if kwargs.pretrain:
        kwargs.exp_name = 'pre-train'
        if kwargs.dataset.batch_size is None:
            kwargs.dataset.batch_size = kwargs.batch_size_map.get(kwargs.exp_name, 32)  # fallback default        
            print('Using default batch_size set to: ', kwargs.dataset.batch_size)
        for seed in kwargs.seeds:   
            pl.seed_everything(seed)    
            kwargs.seed = seed
            kwargs.checkpoint = fix_default_checkpoint(kwargs)
            print("Pretraining model...")
            pretrain(kwargs, wandb_logger)
        
    elif kwargs.test:
        print("Testing model...")        
        #for seed in kwargs.seeds:   
        if 'competition' in exp_name:    
            kwargs.exp_name = 'competition'                        
            for method in kwargs.methods:       
                if kwargs.only_test:
                    print(f'Using method: {method}')             
                pl.seed_everything(seed)       
                kwargs.seed = seed
                kwargs.checkpoint.seed = seed
                kwargs.method = method                    
                test(kwargs)
        elif 'calibrate' in exp_name:    
            kwargs.exp_name = 'calibrate'
            pl.seed_everything(seed)       
            kwargs.seed = seed
            kwargs.checkpoint.seed = seed
            test(kwargs)     
        elif 'quantize' in exp_name:    
            kwargs.exp_name = 'quantize'
            for slot in kwargs.models.slots:
                kwargs.models.S = slot
                repr_dim = 768 if model_class in ['vit', "convnext"] else 2048
                kwargs.models.d = int(repr_dim/slot)
                kwargs.dataset.feature_dim = repr_dim
                print(f'Testing model with {kwargs.models.d} dimensions per slot...')
                print(f'Testing model with {kwargs.models.S} slots...')
                for kappa in kwargs.models.kappas:
                    kwargs.models.K = kappa               
                    print(f'Testing model with {kwargs.models.K} codewords...')
                    pl.seed_everything(seed)       
                    kwargs.seed = seed
                    kwargs.checkpoint.seed = seed
                    test(kwargs)    
        elif 'pre-train' in exp_name:    
            kwargs.exp_name = 'pre-train'
            for seed in kwargs.seeds:   
                pl.seed_everything(seed)     
                kwargs.seed = seed
                test(kwargs)       
        elif 'replicate' in exp_name:    
            kwargs.exp_name = 'replicate'                            
            pl.seed_everything(seed)       
            kwargs.seed = seed
            kwargs.checkpoint.seed = seed
            test(kwargs)   
        elif 'ess_plot' in exp_name:
            kwargs.exp_name = 'ess_plot'                            
            # pl.seed_everything(seed)       
            # kwargs.seed = seed
            # kwargs.checkpoint.seed = seed
            test(kwargs)     
        elif 'ablate_cal_size' in exp_name:
            kwargs.exp_name = 'ablate_cal_size'                            
            # pl.seed_everything(seed)       
            # kwargs.seed = seed
            # kwargs.checkpoint.seed = seed
            test(kwargs)   
        elif 'ablate_s_k_tissue' in exp_name:
            kwargs.exp_name = 'ablate_s_k_tissue'                            
            # pl.seed_everything(seed)       
            # kwargs.seed = seed
            # kwargs.checkpoint.seed = seed
            test(kwargs)   
        elif 'convnext_results' in exp_name:
            kwargs.exp_name = 'convnext_results'                            
            # pl.seed_everything(seed)       
            # kwargs.seed = seed
            # kwargs.checkpoint.seed = seed
            test(kwargs)   
        elif 'ablate_vq' in exp_name:
            kwargs.exp_name = 'ablate_vq'                            
            # pl.seed_everything(seed)       
            # kwargs.seed = seed
            # kwargs.checkpoint.seed = seed
            test(kwargs)     
        elif 'corruption_results' in exp_name:
            kwargs.exp_name = 'corruption_results'                            
            # pl.seed_everything(seed)       
            # kwargs.seed = seed
            # kwargs.checkpoint.seed = seed
            test(kwargs)  
        elif 'calsize_plots' in exp_name:
            kwargs.exp_name = 'calsize_plots'                            
            # pl.seed_everything(seed)       
            # kwargs.seed = seed
            # kwargs.checkpoint.seed = seed
            test(kwargs)  
        elif 'box_plots' in exp_name:
            kwargs.exp_name = 'box_plots'                            
            # pl.seed_everything(seed)       
            # kwargs.seed = seed
            # kwargs.checkpoint.seed = seed
            test(kwargs)  
        elif 'usage_stats' in exp_name:
            kwargs.exp_name = 'usage_stats'                            
            # pl.seed_everything(seed)       
            # kwargs.seed = seed
            # kwargs.checkpoint.seed = seed
            test(kwargs)              
                      
    elif kwargs.calibrate:                
        kwargs.exp_name = 'calibrate'        
        if kwargs.dataset.batch_size is None:
            kwargs.dataset.batch_size = kwargs.batch_size_map.get(kwargs.exp_name, 512)  # fallback default        
            print('Using default batch_size set to: ', kwargs.dataset.batch_size)
            print(f"Calibrating model with {kwargs.calibration_method} technique...")
        for seed in kwargs.seeds:  
            pl.seed_everything(seed)     
            kwargs.seed = seed
            kwargs.checkpoint.seed = seed 
            for subsample in kwargs.dataset.subsamples:
                kwargs.dataset.subsample = subsample
                print(f'\nUsing subsample: {kwargs.dataset.subsample}\n')        
                calibrate(kwargs, wandb_logger)
            
    elif kwargs.competition:        
        kwargs.exp_name = 'competition'
        if kwargs.dataset.batch_size is None:
            kwargs.dataset.batch_size = kwargs.batch_size_map.get(kwargs.exp_name, 512)  # fallback default        
            print('Using default batch_size set to: ', kwargs.dataset.batch_size)
            print("Testing peroformance of competitors...")
        for seed in kwargs.seeds:            
            pl.seed_everything(seed) 
            kwargs.seed = seed
            kwargs.checkpoint.seed = seed
            for subsample in kwargs.dataset.subsamples:
                kwargs.dataset.subsample = subsample
                print(f'\nUsing subsample: {kwargs.dataset.subsample}\n')        
                competition(kwargs, wandb_logger)
            
    elif kwargs.viz_and_test:
        print("Computing visualisations and computing aggreagting metricss...")                                 
        viz_and_test(kwargs)
        
    elif kwargs.quantize:                
        kwargs.exp_name = 'quantize'

        if kwargs.models.quadratic:
            print("Using quadratic calibration model...")
        else:
            print("Using linear calibration model...")
                
        if kwargs.dataset.batch_size is None:
            kwargs.dataset.batch_size = kwargs.batch_size_map.get(kwargs.exp_name, 512)  # fallback default        
            print('Using default batch_size set to: ', kwargs.dataset.batch_size) 
        else:
            print('Using batch_size set to: ', kwargs.dataset.batch_size)
                       
        for slot in kwargs.models.slots:
            kwargs.models.S = slot
            repr_dim = 768 if model_class in ['vit', "convnext"] else 2048
            # kwargs.models.d = int(repr_dim / slot)
            kwargs.models.d = int(repr_dim/kwargs.models.S)
            # kwargs.dataset.feature_dim = repr_dim
            print(f'Testing model with {kwargs.models.d} dimensions per slot...')
            print(f'Testing model with {kwargs.models.S} slots...')
            print(f'Quantizing model with {kwargs.models.S} slots...')
            
            for kappa in kwargs.models.kappas:
                kwargs.models.K = kappa
                print(f'Quantizing model with {kwargs.models.K} codewords...')
                
                for seed in kwargs.seeds:               
                    pl.seed_everything(seed)     
                    kwargs.seed = seed
                    kwargs.checkpoint.seed = seed
                    for subsample in kwargs.dataset.subsamples:
                        kwargs.dataset.subsample = subsample
                        print(f'\nUsing subsample: {kwargs.dataset.subsample}\n')        
                        quantize(kwargs, wandb_logger)

    elif kwargs.replicate:                
        kwargs.exp_name = 'replicate'        
            
        if kwargs.dataset.batch_size is None:
            kwargs.dataset.batch_size = kwargs.batch_size_map.get(kwargs.exp_name, 512)  # fallback default        
            print('Using default batch_size set to: ', kwargs.dataset.batch_size)                 
                
        for seed in kwargs.seeds: 
            print('Starting replicator calibration')              
            pl.seed_everything(seed)     
            kwargs.seed = seed
            kwargs.checkpoint.seed = seed
            for subsample in kwargs.dataset.subsamples:
                kwargs.dataset.subsample = subsample
                print(f'\nUsing subsample: {kwargs.dataset.subsample}\n')        
                replicate(kwargs, wandb_logger)
    
    
    wandb.finish()
    del wandb_logger

    end = time.time()
    time_elapsed = end-start
    print('Total running time: {:.0f}h {:.0f}m'.
        format(time_elapsed // 3600, (time_elapsed % 3600)//60))
    
    
#@hydra.main(config_path='./src/configs', config_name='config_local', version_base=None)
def main_entry():                    
    
    #cli_overrides = [arg for arg in sys.argv[1:] if "=" in arg]
    excluded_keys = {"dataset", "models"}
    init_overrides = [
        arg for arg in sys.argv[1:]
        if "=" in arg and arg.split(".")[0] not in excluded_keys
    ]
    second_overrides = [
        arg for arg in sys.argv[1:]
        if "=" in arg and arg.split(".")[0] in excluded_keys
    ]
    with initialize(config_path="./src/configs", version_base=None):
                
        cfg = compose(config_name="config_local", overrides=init_overrides)        
        
        dataset_name = cfg.data
        
        if cfg.run_all:


            for stage in cfg.stages:
                print(f"\nPreparing config for stage: {stage}")
                cfg = build_cfg_for_stage(stage, init_overrides, second_overrides, dataset_name, cfg)                

                # activate only the current stage                                
                cfg.pretrain = (stage == "pre-train")
                cfg.calibrate = (stage == "calibrate") or (stage == "kernel_calibrate")
                # cfg.kernel_calibrate = (stage == "kernel_calibrate")
                cfg.quantize = (stage == "quantize")
                cfg.competition = (stage == "competition")
                cfg.replicate = (stage == "replicate")
                cfg.test = False
                cfg.viz_and_test = False
                cfg.run_all = False 
                
                update_config_for_stage(cfg, stage)

                print(cfg)

                main(cfg)             
                
        elif cfg.pretrain:
            model_name = cfg.models_map[cfg.data].strip()            
            full_overrides = init_overrides + [f"dataset={dataset_name}", f"models={model_name}"] + second_overrides
            cfg = compose(config_name="config_local", overrides=full_overrides)
            main(cfg)
            
        elif cfg.calibrate:
            model_name = 'calibrator'
            full_overrides = init_overrides + [f"dataset={dataset_name}", f"models={model_name}"] + second_overrides            
            cfg = compose(config_name="config_local", overrides=full_overrides)
            main(cfg)
        
        elif cfg.quantize:
            model_name = 'quantizer'
            full_overrides = init_overrides + [f"dataset={dataset_name}", f"models={model_name}"] + second_overrides            
            cfg = compose(config_name="config_local", overrides=full_overrides)      
            main(cfg)      
            
        elif cfg.replicate:
            model_name = 'replicator'
            full_overrides = init_overrides + [f"dataset={dataset_name}", f"models={model_name}"] + second_overrides            
            cfg = compose(config_name="config_local", overrides=full_overrides)
            main(cfg)
            
        elif cfg.test:
            if cfg.exp_name not in ['pre-train', 'calibrate', 'competition', 'quantize', 'replicate', 'ess_plot', 'ablate_cal_size', 'ablate_s_k_tissue', 'convnext_results', 'ablate_vq', 'corruption_results', 'calsize_plots', 'box_plots', 'usage_stats']:
                raise ValueError(f"Explicitly provide 'exp_name' argument from CLI when testing! Allowed values are 'pre-train', 'calibrate', 'competition', 'quantize', 'replicate'. Instead '{cfg.exp_name}' was given!")                 
            
            elif cfg.exp_name == 'pre-train':
                model_name = cfg.models_map[cfg.data].strip() 
                full_overrides = init_overrides + [f"dataset={dataset_name}", f"models={model_name}"] + second_overrides
                model_name = cfg.models_map[cfg.data]
                cfg = compose(config_name="config_local", overrides=full_overrides)
                
            elif cfg.exp_name == 'calibrate':
                model_name = 'calibrator'
                full_overrides = init_overrides + [f"dataset={dataset_name}", f"models={model_name}"] + second_overrides
                model_name = 'calibrator'
                cfg = compose(config_name="config_local", overrides=full_overrides)
                
            elif cfg.exp_name == 'quantize':
                model_name = 'quantizer'
                full_overrides = init_overrides + [f"dataset={dataset_name}", f"models={model_name}"] + second_overrides
                model_name = 'quantizer'
                cfg = compose(config_name="config_local", overrides=full_overrides)
            
            elif cfg.exp_name == 'replicate':
                model_name = 'replicator'
                full_overrides = init_overrides + [f"dataset={dataset_name}", f"models={model_name}"] + second_overrides
                model_name = 'replicator'
                cfg = compose(config_name="config_local", overrides=full_overrides)            
                
            elif cfg.exp_name == 'competition':
                model_name = 'competition'
                full_overrides = init_overrides + [f"dataset={dataset_name}", f"models={model_name}"] + second_overrides
                model_name = 'competition'
                cfg = compose(config_name="config_local", overrides=full_overrides)
                
            elif cfg.exp_name == 'ess_plot':
                model_name = 'competition'
                full_overrides = init_overrides + [f"dataset={dataset_name}", f"models={model_name}"] + second_overrides
                model_name = 'competition'
                cfg = compose(config_name="config_local", overrides=full_overrides)
                
            elif cfg.exp_name == 'ablate_cal_size':
                model_name = 'competition'
                full_overrides = init_overrides + [f"dataset={dataset_name}", f"models={model_name}"] + second_overrides
                model_name = 'competition'
                cfg = compose(config_name="config_local", overrides=full_overrides)
            
            elif cfg.exp_name == 'ablate_s_k_tissue':
                model_name = 'competition'
                full_overrides = init_overrides + [f"dataset={dataset_name}", f"models={model_name}"] + second_overrides
                model_name = 'competition'
                cfg = compose(config_name="config_local", overrides=full_overrides)
                
            elif cfg.exp_name == 'convnext_results':
                model_name = 'competition'
                full_overrides = init_overrides + [f"dataset={dataset_name}", f"models={model_name}"] + second_overrides
                model_name = 'competition'
                cfg = compose(config_name="config_local", overrides=full_overrides)
            
            elif cfg.exp_name == 'ablate_vq':
                model_name = 'competition'
                full_overrides = init_overrides + [f"dataset={dataset_name}", f"models={model_name}"] + second_overrides
                model_name = 'competition'
                cfg = compose(config_name="config_local", overrides=full_overrides)

            elif cfg.exp_name == 'corruption_results':
                model_name = 'competition'
                full_overrides = init_overrides + [f"dataset={dataset_name}", f"models={model_name}"] + second_overrides
                model_name = 'competition'
                cfg = compose(config_name="config_local", overrides=full_overrides)
            
            elif cfg.exp_name == 'calsize_plots':
                model_name = 'competition'
                full_overrides = init_overrides + [f"dataset={dataset_name}", f"models={model_name}"] + second_overrides
                model_name = 'competition'
                cfg = compose(config_name="config_local", overrides=full_overrides)
            
            elif cfg.exp_name == 'box_plots':
                model_name = 'competition'
                full_overrides = init_overrides + [f"dataset={dataset_name}", f"models={model_name}"] + second_overrides
                model_name = 'competition'
                cfg = compose(config_name="config_local", overrides=full_overrides)
                
            elif cfg.exp_name == 'usage_stats':
                model_name = 'competition'
                full_overrides = init_overrides + [f"dataset={dataset_name}", f"models={model_name}"] + second_overrides
                model_name = 'competition'
                cfg = compose(config_name="config_local", overrides=full_overrides)
                
            main(cfg)
                
        elif cfg.competition: #.exp_name == 'competition':
            model_name = 'competition'
            full_overrides = init_overrides + [f"dataset={dataset_name}", f"models={model_name}"] + second_overrides
            cfg = compose(config_name="config_local", overrides=full_overrides)
            main(cfg)
            
        elif cfg.viz_and_test: #.exp_name == 'competition':
            model_name = 'competition'
            full_overrides = init_overrides + [f"dataset={dataset_name}", f"models={model_name}"] + second_overrides
            cfg = compose(config_name="config_local", overrides=full_overrides)
            main(cfg)                                        
        
    
##################################################################################################################

def run_pretrain(kwargs, wandb_logger):
    kwargs.exp_name = 'pre-train'
    if kwargs.dataset.batch_size is None:
        kwargs.dataset.batch_size = kwargs.batch_size_map.get(kwargs.exp_name, 32)  # fallback default        
        print('Using default batch_size set to: ', kwargs.dataset.batch_size)
    for seed in kwargs.seeds:   
        pl.seed_everything(seed)    
        kwargs.seed = seed
        kwargs.checkpoint = fix_default_checkpoint(kwargs)        
        print("Pretraining model...")
        pretrain(kwargs, wandb_logger)
        
def run_calibrate(kwargs, wandb_logger):            
    kwargs.exp_name = 'calibrate'        
    if kwargs.dataset.batch_size is None:
        kwargs.dataset.batch_size = kwargs.batch_size_map.get(kwargs.exp_name, 512)  # fallback default        
        print('Using default batch_size set to: ', kwargs.dataset.batch_size)
        print(f"Calibrating model with {kwargs.calibration_method} technique...")
    for seed in kwargs.seeds:  
        pl.seed_everything(seed)     
        kwargs.seed = seed
        kwargs.checkpoint.seed = seed 
        for subsample in kwargs.dataset.subsamples:
            kwargs.dataset.subsample = subsample
            print(f'\nUsing subsample: {kwargs.dataset.subsample}\n')        
            calibrate(kwargs, wandb_logger)
            
def run_competition(kwargs, wandb_logger):      
    kwargs.exp_name = 'competition'
    if kwargs.dataset.batch_size is None:
        kwargs.dataset.batch_size = kwargs.batch_size_map.get(kwargs.exp_name, 512)  # fallback default        
        print('Using default batch_size set to: ', kwargs.dataset.batch_size)
        print("Testing peroformance of competitors...")
    for seed in kwargs.seeds:            
        pl.seed_everything(seed) 
        kwargs.seed = seed
        kwargs.checkpoint.seed = seed
        for subsample in kwargs.dataset.subsamples:
            kwargs.dataset.subsample = subsample
            print(f'\nUsing subsample: {kwargs.dataset.subsample}\n')        
            competition(kwargs, wandb_logger)                
        
def run_quantize(kwargs, wandb_logger, model_class=None):
    kwargs.exp_name = 'quantize'

    if kwargs.models.quadratic:
        print("Using quadratic calibration model...")
    else:
        print("Using linear calibration model...")
            
    if kwargs.dataset.batch_size is None:
        kwargs.dataset.batch_size = kwargs.batch_size_map.get(kwargs.exp_name, 512)  # fallback default        
        print('Using default batch_size set to: ', kwargs.dataset.batch_size) 
    else:
        print('Using batch_size set to: ', kwargs.dataset.batch_size)
                    
    for slot in kwargs.models.slots:
        kwargs.models.S = slot
        repr_dim = 768 if model_class in ['vit', "convnext"] else 2048
        # kwargs.models.d = int(repr_dim / slot)
        kwargs.models.d = int(repr_dim/kwargs.models.S)
        # kwargs.dataset.feature_dim = repr_dim
        print(f'Testing model with {kwargs.models.d} dimensions per slot...')
        print(f'Testing model with {kwargs.models.S} slots...')
        print(f'Quantizing model with {kwargs.models.S} slots...')
        
        for kappa in kwargs.models.kappas:
            kwargs.models.K = kappa
            print(f'Quantizing model with {kwargs.models.K} codewords...')
            
            for seed in kwargs.seeds:               
                pl.seed_everything(seed)     
                kwargs.seed = seed
                kwargs.checkpoint.seed = seed
                for subsample in kwargs.dataset.subsamples:
                    kwargs.dataset.subsample = subsample
                    print(f'\nUsing subsample: {kwargs.dataset.subsample}\n')        
                    quantize(kwargs, wandb_logger)

def run_replicate(kwargs, wandb_logger):
    kwargs.exp_name = 'replicate'        
        
    if kwargs.dataset.batch_size is None:
        kwargs.dataset.batch_size = kwargs.batch_size_map.get(kwargs.exp_name, 512)  # fallback default        
        print('Using default batch_size set to: ', kwargs.dataset.batch_size)                 
            
    for seed in kwargs.seeds: 
        print('Starting replicator calibration')              
        pl.seed_everything(seed)     
        kwargs.seed = seed
        kwargs.checkpoint.seed = seed
        for subsample in kwargs.dataset.subsamples:
            kwargs.dataset.subsample = subsample
            print(f'\nUsing subsample: {kwargs.dataset.subsample}\n')        
            replicate(kwargs, wandb_logger)
    
def run_stage(kwargs, stage_name, wandb_logger, model_class):
    if stage_name == "pre-train":
        run_pretrain(kwargs, wandb_logger)
    elif stage_name == "calibrate" or stage_name == "kernel_calibrate":
        run_calibrate(kwargs, wandb_logger)
    elif stage_name == "quantize":
        run_quantize(kwargs, wandb_logger, model_class)
    elif stage_name == "competition":
        run_competition(kwargs, wandb_logger)
    else:
        raise ValueError(f"Unknown stage: {stage_name}")

def build_cfg_for_stage(stage, init_overrides, second_overrides, dataset_name, base_cfg):
    if stage == "pre-train":
        model_name = base_cfg.models_map[dataset_name].strip()
    elif stage == "calibrate" or stage == "kernel_calibrate" :
        model_name = "calibrator"
    elif stage == "quantize":
        model_name = "quantizer"
    elif stage == "replicate":
        model_name = "replicator"
    elif stage == "competition":
        model_name = "competition"
    else:
        raise ValueError(f"Unknown stage: {stage}")        

    full_overrides = init_overrides + [f"dataset={dataset_name}", f"models={model_name}"] + second_overrides
    return compose(config_name="config_local", overrides=full_overrides)

def update_config_for_stage(cfg, stage):

    
    ################### LCN AND KERNEL CONFIGS ###################
    if stage == 'calibrate' or stage == 'kernel_calibrate':
        
        cfg.dataset.batch_size=1024
        cfg.extract_embeddings = False

        # calibration hyper-parameters                                                                                                                                                                                        
        cfg.models.log_var_initializer = 10.                                   # Control initial variance. Default 10. Used 6 for convnext. In general less classes -> smaller values. If sampling is true set to small values to initialize encodings close to inputs   #   
        cfg.models.hidden_dim = 64                                             # Size of hidden layer of calibrating net                                                                                                     #
        cfg.models.alpha1 = 1.                                                 # Classification contraint penalty                                                                                                            #
        cfg.models.alpha2 = 0.                                                 # Switch-off variance penalty                                                                                                                 #
        cfg.models.lambda_kl = 1. if stage == 'calibrate' else 0.              # JSD hyperparameter                                                                                                                #
        cfg.models.entropy_factor = 0.                                         # Control peakness of entropy of model confidence.                                                                                            #
        cfg.models.epochs = 100                                                # Number of epochs                                                                                                                            #
        cfg.models.noise = 0.                                                  # Std of normal distribution to sample noise to add to input logits to improve generalisation                                                 #
        cfg.models.smoothing = 0.                                              # Random label-smootihng parameter to improve on sparse data (many classes -> more sparse matrices!) 4.5e-2                                   #
        cfg.models.logits_scaling = 1.                                         # Scaling logits to make softmax more picked for stable gumbel-softmax sampling of argmax to ensure calssification constraint                 #
        cfg.models.sampling = False                                            # If true uses parametrised varances to to add noise to latent representations via reparametrisation trick                                    #
        cfg.models.use_empirical_freqs = True                                  # If true computes cross-entropy with empirical frequencies about of each point in place else uses softmax of new latent encodings            #
        cfg.models.predict_labels = True                                       # If true computes cross-entropy with true labels else uses predicted labels of frozen model                                                  #
        cfg.models.js_distance = True                                          # If true copmutes JS distance in place of kl divergence between model scroes and kernel estimates                                            #
        cfg.models.interpolation_epochs = 0.                                   # Number of epochs for lambda_kl to drop to alpha1. To not interolate set as 0                                                                #
        cfg.models.dropout = 0.3                                               # Dropout to regularize calibrator

        cfg.models.augment = False              # If true uses data augmentation during training of calibrator
        cfg.models.adabw = False                # If true uses adaptive bandwidth for lce computation else fixed bandwidth
        cfg.models.fixed_var = True
        cfg.models.linearly_combine_pca = True  # If true learns a linear combination of pca components to add to similarity embeddings else adds pca components directly

        cfg.models.kernel_only = False          # If true uses only kernel density estimates for calibration else combines with base model predictions

        cfg.models.alpha_sim = 1.0              # Initial similarity scaling factor if linearly_combine_pca is true
        cfg.models.alpha_cls = 1.0              # Initial class scaling factor if linearly_combine_pca is true
                                                                                                    

    ################### QUANTIZER CONFIGS ###################
    elif stage == 'quantize':
        
        cfg.dataset.batch_size=128
        cfg.extract_embeddings = False
  
        cfg.models.optimizer.weight_decay = 1e-3 #0.0001                
        
        # calibration hyper-parameters                                                                                                                                                                                        
        cfg.models.slots = [64] #[16, 32, 128, 256] # USED 128 FOR VIT WITH TISSUE # 192 FOR VIT WITH CIFAR100
        cfg.models.S = 64
        cfg.models.d = 32
        cfg.models.kappas = [64] #[128] #[32, 64] #[16, 32, 128, 256] # USED 32 FOR VIT WITH CIFR100 and for TISSUE RESNET
        cfg.models.K = 64 #[64] #tissue: 16 # 32 64?
        cfg.models.vq_decay = 0.99 # 0.5
        cfg.models.vq_eps = 1e-8
        cfg.models.hidden = 0.
        cfg.models.dropout = 0.1
        cfg.models.epochs = 100
        cfg.models.similarity_dim = 50
        cfg.models.adabw = False                # If true uses adaptive bandwidth for lce computation else fixed bandwidth
        cfg.models.learn_pi = True              # If true learns the prior probabilities of the VQ codebook vectors
        cfg.models.diag = False                 # If true learns the diagonal latent miscalibration component
        cfg.models.lambda_reg = 0.              # Weight of the regularization term (not used?)
        cfg.models.random = False               # If true uses random quantization indices instead of VQ indices during training
        cfg.models.L1 = False                   # If true uses L1 distance instead of L2 distance for the VQ codebook vectors
        cfg.models.quantization_only = False    # If true only performs quantization without calibration
        cfg.models.standard_dirichlet = False   # If true uses standard dirichlet calibration on top of quantization
        cfg.models.learn_bias = False           # If true learns a bias term per codebook vector
        cfg.models.quadratic = False            # If true uses quadratic calibration on top of quantization

    ################### COMPETITION CONFIGS ###################
    elif stage == 'competition':
        cfg.dataset.batch_size=128
        
        cfg.models.max_iter = 2000
        cfg.models.temp_lr = 1e-3 # 0.01
        cfg.models.adabw = False
        cfg.models.num_neighbors = 50        
    
    
    elif stage == "pre-train":
        cfg.dataset.batch_size=128        
        cfg.models.epochs = cfg.checkpoint.epochs
        cfg.extract_embeddings = True
        
##################################################################################################################        
        
        
if __name__ == "__main__":
    main_entry()      
    
    
    
    
