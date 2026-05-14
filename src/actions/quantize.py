import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from algorithms.networks.networks import *
from algorithms.trainers.trainers import *
import hydra
from hydra import initialize, compose
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, open_dict, OmegaConf
import time
from utils.utils import *
from data_sets.dataset import *
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from actions.test import test
from calibrator.cal_trainer import *
from calibrator.local_net import *
from calibrator.quant_trainer import *
from calibrator.quantisation_head import *
from calibrator.VQCAL_trainer import *
import optuna
from optuna.samplers import NSGAIISampler
from hp_opt.hp_opt import *
from pytorch_lightning.loggers import WandbLogger
import json
from tqdm import tqdm

def quantize(kwargs, wandb_logger):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    seed = kwargs.seed
    total_epochs = kwargs.models.epochs    
    cuda_device = kwargs.cuda_device
    #pl.seed_everything(seed, workers=True)  
    
    if kwargs.data == 'synthetic':
        dataset = SynthData(kwargs, experiment=kwargs.exp_name)  
    elif kwargs.data == 'covtype':
        dataset = CovTypeData(kwargs, experiment=kwargs.exp_name)  
    elif kwargs.data == 'otto':
        dataset = OttoData(kwargs, experiment=kwargs.exp_name)                    
    elif kwargs.data == 'mnist':
        if kwargs.dataset.variant:
            kwargs.data = kwargs.data + '_' + kwargs.dataset.variant                        
        dataset = MnistData(kwargs, experiment=kwargs.exp_name)
    elif kwargs.data == 'weather':
        dataset = WeatherData(kwargs, experiment=kwargs.exp_name)                
    elif kwargs.data == 'tissue':
        dataset = MedMnistData(kwargs, experiment=kwargs.exp_name)   
    elif kwargs.data == 'path':
        dataset = MedMnistData(kwargs, experiment=kwargs.exp_name)       
    elif kwargs.data == 'cifar10':
        dataset = Cifar10Data(kwargs, experiment=kwargs.exp_name)
    elif kwargs.data == 'cifar10_ood':
        dataset = Cifar10OODData(calibration=kwargs.calibration)
    elif kwargs.data == 'cifar10LT':
        dataset = Cifar10LongTailData(kwargs, experiment=kwargs.exp_name)
    elif kwargs.data == 'cifar100':
        dataset = Cifar100Data(kwargs, experiment=kwargs.exp_name) 
    elif kwargs.data == 'cifar100_longtail':
        dataset = Cifar100LongTailData(calibration=kwargs.calibration)
    elif kwargs.data == 'Imagenet':
        dataset = ImagenetData(calibration=kwargs.calibration)
    elif kwargs.data == 'imagenet_ood':
        dataset = ImagenetOODData(calibration=kwargs.calibration)
    elif kwargs.data == 'imagenet_longtail':
        dataset = ImagenetLongTailData(calibration=kwargs.calibration)    
        
    corruptions = [
        "gaussian_noise",
        "shot_noise",
        "impulse_noise",
        "defocus_blur",
        "glass_blur",
        "motion_blur",
        "zoom_blur",
        "fog",
        "snow",
        "frost", # try        
        "brightness", # good
        "contrast",
        "pixelate",        
    ]        
    
    if (kwargs.corruption_type) and (kwargs.corruption_type not in corruptions):
        raise ValueError(f'Unknown corruption type! {kwargs.corruption_type} was given.')
    
    sev = kwargs.severity
    if kwargs.corruption_type:
        corruption_text = kwargs.corruption_type
        corruption_name = kwargs.corruption_type
        if sev > 0:
            severity = kwargs.severity
            corruption_text += f'_severity_{severity}'
        print("CORRUPTION TEXT: ", corruption_text)
    else:
        corruption_text = "None"
    
    epochs = kwargs.checkpoint.epochs
    if epochs == 9:
        model_class = 'resnet'
    elif kwargs.checkpoint.epochs == 5:
        model_class = 'vit'
    elif kwargs.checkpoint.epochs == 20:
        model_class = 'convnext'
    else: # ftt uses 50 
        model_class = 'ftt'
        if not kwargs.data == 'weather':
            raise ValueError(
                f'Checkpoint not corresponding to a trained modl! {kwargs.checkpoint.epochs} was given but only 9 and 20 are supported')
                    
    name = kwargs.exp_name
    if kwargs.models.S != 64:
        name += f'slot-{kwargs.models.S}'
    if kwargs.models.K != 64:
        name += f'kappa-{kwargs.models.K}'
    if kwargs.models.random:
        name += '_random'
    if kwargs.models.L1:
        name += '_L1'
    if kwargs.models.quantization_only: 
        name += '_quantonly'
    if kwargs.models.standard_dirichlet:
        name += '_stdcal'    
    if kwargs.models.quadratic:
        name += '_quadratic'
    print(f"\n\n\n\n\n\n\n\n\n FORZA NAPOLI, W {name}! \n\n\n\n\n\n\n\n\n")
    path = f"checkpoints/{name}/{kwargs.data}_{kwargs.dataset.num_classes}_classes_{kwargs.dataset.num_features}_features/"    
    os.makedirs(path, exist_ok=True) 
    
    result_path = f"results/{name}/{kwargs.data}_{kwargs.dataset.num_classes}_classes_{kwargs.dataset.num_features}_features"    
    os.makedirs(result_path, exist_ok=True)    
    
    if kwargs.use_optuna:  
        if kwargs.multi_obj:
            study = optuna.create_study(
                directions=["minimize", "minimize"],  
                sampler=NSGAIISampler(),
                study_name="multi_objective"
            )
            calls = [multi_obj_print_callback]
        else:  
            study = optuna.create_study(direction="minimize", study_name="standard")
            calls = [print_callback]
        study.optimize(
            lambda trial: objective(trial, kwargs, dataset.data_train_cal_loader, dataset.data_val_cal_loader, wandb_logger),
            n_trials=kwargs.n_trials,
            show_progress_bar=True,
            callbacks=calls
        )
        if kwargs.multi_obj:
            fig = optuna.visualization.plot_pareto_front(study)    
            appendix = kwargs.exp_name + '_' + kwargs.data + '_' + f'{kwargs.dataset.num_classes}_classes_' + f'{kwargs.dataset.num_features}_features'        
            fig.write_html(f"results/plots/{appendix}/pare_front.html") #plt.savefig("results/plots/"+ appendix) #fig.write_image(f"results/plots/{kwargs.exp_name}/pareto_front.png")
            pareto_trials = study.best_trials  # List of Pareto-optimal trials
            for trial in pareto_trials:
                print(f"Trial {trial.number}: KL={trial.values[0]}, Constraint={trial.values[1]}")
            best = min(pareto_trials, key=lambda t: t.values[0] + t.values[1])  # Simple sum             
            for key, value in best.params.items():
                print(f"    {key}: {value}")
                kwargs.models[key] = value  
        else:
            # Print best result
            print("Best trial:")
            print(f"  Value: {study.best_trial.value}")
            for key, value in study.best_trial.params.items():
                print(f"    {key}: {value}")
                kwargs.models[key] = value

    ############################# QUANTISATION TRAINING #############################
    pl_model = VQClassifier(kwargs, num_classes=kwargs.dataset.num_classes, feature_dim=kwargs.dataset.feature_dim, feature_loader=dataset.data_train_cal_loader, backbone=model_class)        
    name = kwargs.exp_name
    if kwargs.models.S != 64:
        name += f'slot-{kwargs.models.S}'
    if kwargs.models.K != 64:
        name += f'kappa-{kwargs.models.K}'
    if kwargs.models.random:
        name += '_random'
    if kwargs.models.L1:
        name += '_L1'
    if kwargs.models.quantization_only: 
        name += '_quantonly'
    if kwargs.models.standard_dirichlet:
        name += '_stdcal'    
    if kwargs.models.quadratic:
        name += '_quadratic'
    
    if kwargs.corruption_type:
        raw_results_path_test_cal = "results/{}/{}_{}_classes_{}_features/raw_results_test_quant_corrupt_{}_seed-{}_ep-{}_{}.csv".format(
                name, #kwargs.exp_name,
                kwargs.data,
                kwargs.dataset.num_classes,
                kwargs.dataset.num_features,
                corruption_text, #kwargs.corruption_type,
                seed,
                total_epochs,
                model_class           
            )
        raw_results_path_train_cal = "results/{}/{}_{}_classes_{}_features/raw_results_train_quant_corrupt_{}_seed-{}_ep-{}_{}.csv".format(
            name, #kwargs.exp_name,
            kwargs.data,
            kwargs.dataset.num_classes,
            kwargs.dataset.num_features,
            corruption_text, #kwargs.corruption_type,
            seed,
            total_epochs,                       
            model_class
        )
    else:    
        raw_results_path_test_cal = "results/{}/{}_{}_classes_{}_features/raw_results_test_quant_seed-{}_ep-{}_{}.csv".format(
                name, #kwargs.exp_name,
                kwargs.data,
                kwargs.dataset.num_classes,
                kwargs.dataset.num_features,
                seed,
                total_epochs,
                model_class           
            )
        if kwargs.data == 'weather' and kwargs.dataset.shift:
            raw_results_path_test_cal = "results/{}/{}_{}_classes_{}_features/raw_results_test_quant_shift_seed-{}_ep-{}_{}.csv".format(
                name, #kwargs.exp_name,
                kwargs.data,
                kwargs.dataset.num_classes,
                kwargs.dataset.num_features,
                seed,
                total_epochs,
                model_class           
            )
        raw_results_path_train_cal = "results/{}/{}_{}_classes_{}_features/raw_results_train_quant_seed-{}_ep-{}_{}.csv".format(
            name, #kwargs.exp_name,
            kwargs.data,
            kwargs.dataset.num_classes,
            kwargs.dataset.num_features,
            seed,
            total_epochs,
            model_class                       
        )
    
    if (kwargs.extract_embeddings): #(kwargs.corruption_type) or 
        best_model_path = path + f"VQHEAD_seed-{seed}_ep-{total_epochs}_{model_class}.ckpt"
        print(F'LOADING CHECKPOINT FILE {best_model_path}')
        checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
        # import pdb; pdb.set_trace()
        state_dict = checkpoint["state_dict"]
        print(state_dict.keys())
        pl_model.load_state_dict(state_dict)
    else:
        print(F'BEGIN QUANTISATION FOR {total_epochs} EPOCHS WITH SEED {seed}!')        
        trainer = pl.Trainer(
                max_epochs=total_epochs,
                accelerator="cuda",
                devices=[cuda_device],
                logger=wandb_logger,
                check_val_every_n_epoch=1,
                #gradient_clip_val=5,
                deterministic=False,
                callbacks=[ #CalibrationPlotCallback(kwargs, dataset.data_train_cal_loader, every_n_epochs=1, device="cuda", type='train'), 
                            #CalibrationPlotCallback(kwargs, dataset.data_test_cal_loader, every_n_epochs=1, device="cuda", type='test'),
                            EarlyStopping(monitor="val_loss", #val_kl
                                        patience=10, #5
                                        mode="min", 
                                        verbose=True, 
                                        min_delta=0.0),
                            ModelCheckpoint(monitor="val_loss", #val_kl                                                                                               # Metric to track
                                mode="min",                                                                                                     # Lower is better
                                save_top_k=1,                                                                                                   # Only keep the best model
                                filename=f"VQHEAD_seed-{seed}_ep-{total_epochs}_{model_class}",                                                           # Static filename (no epoch suffix)
                                dirpath=path,                                                                                                   # Save in your existing checkpoint folder
                                save_weights_only=True,                                                                                         # Save only weights (not full LightningModule)
                                auto_insert_metric_name=False,                                                                                  # Prevent metric name in filename
                                every_n_epochs=1,                                                                                               # Run every epoch                    
                                enable_version_counter=False,
                                verbose=True
                            ) 
                ]
        )   
        start = time.time()
        trainer.fit(pl_model, dataset.data_train_cal_loader, #train
                        dataset.data_val_cal_loader)
        train_time = time.time() - start
        print(train_time)
        
        # path_model = join(path, f"classifier_seed-{seed}_ep-{total_epochs}")
        # torch.save(pl_model.model.state_dict(), path_model)
        best_model_path = trainer.checkpoint_callback.best_model_path
        
        print(F'LOADING CHECKPOINT FILE {best_model_path}')
        checkpoint = torch.load(best_model_path, weights_only=False, map_location="cpu")
        pl_model.load_state_dict(checkpoint["state_dict"], strict=True)
        
    vq_classifier = pl_model
    # checkpoint = torch.load(best_model_path)
    # pl_model.load_state_dict(checkpoint['state_dict'])
    # path_model = join(path, f"classifier_seed-{seed}_ep-{total_epochs}")
    # torch.save(pl_model.state_dict(), path_model)
    # print("SAVED MODEL TO ", path_model)
    print("QUANTISATION OVER!")
    
    ############################# VQ CALIBRATION #############################
    
    pl_model = VQCalibrator(vq_classifier, kwargs)
    name = kwargs.exp_name
    if kwargs.models.S != 64:
        name += f'slot-{kwargs.models.S}'
    if kwargs.models.K != 64:
        name += f'kappa-{kwargs.models.K}'
    if kwargs.models.random:
        name += '_random'
    if kwargs.models.L1:
        name += '_L1'
    if kwargs.models.quantization_only: 
        name += '_quantonly'
    if kwargs.models.standard_dirichlet:
        name += '_stdcal'    
    if kwargs.models.quadratic:
        name += '_quadratic'
    
    if kwargs.corruption_type:
        raw_results_path_test_cal = "results/{}/{}_{}_classes_{}_features/raw_results_test_calquant_corrupt_{}_seed-{}_ep-{}_{}.csv".format(
                name, #kwargs.exp_name,
                kwargs.data,
                kwargs.dataset.num_classes,
                kwargs.dataset.num_features,
                corruption_text, #kwargs.corruption_type,
                seed,
                total_epochs,
                model_class           
            )
        raw_results_path_train_cal = "results/{}/{}_{}_classes_{}_features/raw_results_train_calquant_corrupt_{}_seed-{}_ep-{}_{}.csv".format(
            name, #kwargs.exp_name,
            kwargs.data,
            kwargs.dataset.num_classes,
            kwargs.dataset.num_features,
            corruption_text, #kwargs.corruption_type,
            seed,
            total_epochs,   
            model_class                    
        )        
    else: 
        raw_results_path_test_cal = "results/{}/{}_{}_classes_{}_features/raw_results_test_calquant_seed-{}_ep-{}_{}.csv".format(
                name, #kwargs.exp_name,
                kwargs.data,
                kwargs.dataset.num_classes,
                kwargs.dataset.num_features,
                seed,
                total_epochs,           
                model_class
            )
        if kwargs.data == 'weather' and kwargs.dataset.shift:
            raw_results_path_test_cal = "results/{}/{}_{}_classes_{}_features/raw_results_test_calquant_shift_seed-{}_ep-{}_{}.csv".format(
                name, #kwargs.exp_name,
                kwargs.data,
                kwargs.dataset.num_classes,
                kwargs.dataset.num_features,
                seed,
                total_epochs,           
                model_class
            )
        raw_results_path_train_cal = "results/{}/{}_{}_classes_{}_features/raw_results_train_calquant_seed-{}_ep-{}_{}.csv".format(
            name, #kwargs.exp_name,
            kwargs.data,
            kwargs.dataset.num_classes,
            kwargs.dataset.num_features,
            seed,
            total_epochs,
            model_class                       
        )
        
    if (kwargs.extract_embeddings): # kwargs.corruption_type or 
        best_model_path = path + f"VQCALIBRATOR_seed-{seed}_ep-{total_epochs}_{model_class}.ckpt"
        print(F'LOADING CHECKPOINT FILE {best_model_path}')
        checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)        
        # import pdb; pdb.set_trace()
        state_dict = checkpoint["state_dict"]
        print(state_dict.keys())
        pl_model.load_state_dict(state_dict)
    else:
        print(F'BEGIN VQCALIBRATION FOR {total_epochs} EPOCHS WITH SEED {seed}!')        
        trainer = pl.Trainer(
                max_epochs=total_epochs,
                accelerator="cuda",
                devices=[cuda_device],
                logger=wandb_logger,
                check_val_every_n_epoch=1,
                #gradient_clip_val=5,
                deterministic=False,
                callbacks=[ #CalibrationPlotCallback(kwargs, dataset.data_train_cal_loader, every_n_epochs=1, device="cuda", type='train'), 
                            #CalibrationPlotCallback(kwargs, dataset.data_test_cal_loader, every_n_epochs=1, device="cuda", type='test'),
                            EarlyStopping(monitor="cal_val_loss", #cal_val_loss val_kl
                                        patience=10, #5
                                        mode="min", #max
                                        verbose=True, 
                                        min_delta=0.0),
                            ModelCheckpoint(monitor="cal_val_loss", #cal_val_loss val_kl                                                                                               # Metric to track
                                mode="min",  #max                                                                                                   # Lower is better
                                save_top_k=1,                                                                                                   # Only keep the best model
                                filename=f"VQCALIBRATOR_seed-{seed}_ep-{total_epochs}_{model_class}",                                                           # Static filename (no epoch suffix)
                                dirpath=path,                                                                                                   # Save in your existing checkpoint folder
                                save_weights_only=True,                                                                                         # Save only weights (not full LightningModule)
                                auto_insert_metric_name=False,                                                                                  # Prevent metric name in filename
                                every_n_epochs=1,                                                                                               # Run every epoch                    
                                enable_version_counter=False,
                                verbose=True
                            ) 
                ]
        )   
        start = time.time()
        trainer.fit(pl_model, dataset.data_train_cal_loader,
                        dataset.data_val_cal_loader)
        train_time = time.time() - start
        print(train_time)
        
        # path_model = join(path, f"classifier_seed-{seed}_ep-{total_epochs}")
        # torch.save(pl_model.model.state_dict(), path_model)
        best_model_path = trainer.checkpoint_callback.best_model_path
    
        print(F'LOADING CHECKPOINT FILE {best_model_path}')
        checkpoint = torch.load(best_model_path, weights_only=False, map_location="cpu")
        pl_model.load_state_dict(checkpoint["state_dict"], strict=True)
    
    # checkpoint = torch.load(best_model_path)
    # pl_model.load_state_dict(checkpoint['state_dict'])
    # path_model = join(path, f"classifier_seed-{seed}_ep-{total_epochs}")
    # torch.save(pl_model.state_dict(), path_model)
    # print("SAVED MODEL TO ", path_model)    
    print('CALIBRATION OVER')
    #pl_model.print_calibrator_params()
    
    raws = []
    pl_model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pl_model.to(device)

    with torch.no_grad():
        for batch in tqdm(dataset.data_train_cal_loader, desc="Extracting pca features"): #tqdm(dataset.data_train_cal_loader, desc="Extracting pca features"):
            if kwargs.dataset.batch_size > 256:
                # Process in smaller mini-batches to avoid OOM
                mini_batch_size = 256
                num_samples = batch[0].size(0)
                for start_idx in range(0, num_samples, mini_batch_size):
                    end_idx = min(start_idx + mini_batch_size, num_samples)
                    mini_batch = [b[start_idx:end_idx].to(device) for b in batch]
                    raw = pl_model.extract(mini_batch)
                    raws.append(raw)                
            else:    
                batch = [b.to(device) for b in batch]                
                raw = pl_model.extract(batch)
                raws.append(raw)

    #all_raws = torch.cat(all_raws)
    #print('pca shape: ', raws[1]['features'].shape)
    res, pca = get_raw_res(raws, features=True, adabw=kwargs.models.adabw, reduced_dim=kwargs.similarity_dim, quantize=True)
    
    # DO NOT SAVE FOR SPACE
    # res.to_csv(raw_results_path_train_cal, index=False)

    raws = []
    pl_model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pl_model.to(device)

    with torch.no_grad():
        for batch in tqdm(dataset.data_test_cal_loader, desc="Extracting pca features"):
            if kwargs.dataset.batch_size > 256:
                # Process in smaller mini-batches to avoid OOM
                mini_batch_size = 256
                num_samples = batch[0].size(0)
                for start_idx in range(0, num_samples, mini_batch_size):
                    end_idx = min(start_idx + mini_batch_size, num_samples)
                    mini_batch = [b[start_idx:end_idx].to(device) for b in batch]
                    raw = pl_model.extract(mini_batch)
                    raws.append(raw)                
            else:    
                batch = [b.to(device) for b in batch]                
                raw = pl_model.extract(batch)
                raws.append(raw)

    #all_raws = torch.cat(all_raws)
    #print('pca shape: ', raws[1]['features'].shape)
    res, pca = get_raw_res(raws, features=True, adabw=False, reduced_dim=kwargs.similarity_dim, fit_pca=pca, quantize=True) 
    
    #raws = trainer.predict(pl_model, dataset.data_test_cal_loader)
    #res = get_raw_res(raws)
    res.to_csv(raw_results_path_test_cal, index=False)

    print("VQCALIBRATION OVER!")
    
    # Save hp to JSON file
    path = f"hyperparams/{kwargs.data}_{kwargs.checkpoint.num_classes}_classes_{kwargs.checkpoint.num_features}_features"
    os.makedirs(path, exist_ok=True)    
    path_hp = join(path, f"classifier_seed-{seed}_ep-{total_epochs}.json")        
    hp_dict = OmegaConf.to_container(kwargs.models, resolve=True)    
    with open(path_hp, "w") as f:
        json.dump(hp_dict, f, indent=4)

    #print(pl_model.cal.A_code.weight)
    #print(pl_model.cal.B_code.weight)
    print("\nSTART TESTING!")     

    test(kwargs)

    
 
    