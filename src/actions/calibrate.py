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
import optuna
from optuna.samplers import NSGAIISampler
from hp_opt.hp_opt import *
from pytorch_lightning.loggers import WandbLogger
import json
from tqdm import tqdm

def calibrate(kwargs, wandb_logger):
    
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
    
    if kwargs.models.lambda_kl == 0:
        print("PROCEED SAFELY WITH REFERENCE KERNEL CALIBRATION!") # k-cal
    if kwargs.models.kernel_only:
        print("PROCEED SAFELY WITH KERNEL ONLY CALIBRATION!")
        
    if kwargs.data == 'synthetic':
        path = f"checkpoints/{kwargs.exp_name}/{kwargs.data}_{kwargs.checkpoint.num_classes}_classes_{kwargs.checkpoint.num_features}_features"
        os.makedirs(path, exist_ok=True)    
        os.makedirs(f"results/{kwargs.exp_name}/{kwargs.data}_{kwargs.checkpoint.num_classes}_classes_{kwargs.checkpoint.num_features}_features", exist_ok=True)    
        
        pl_model = AuxTrainer(kwargs.models, num_classes=kwargs.checkpoint.num_classes)    

        raw_results_path_test_cal = "results/{}/{}_{}_classes_{}_features/raw_results_test_cal_seed-{}_ep-{}.csv".format(
                kwargs.exp_name,
                kwargs.data,
                kwargs.checkpoint.num_classes,
                kwargs.checkpoint.num_features,
                seed,
                total_epochs           
            )        
    else:        
        path = f"checkpoints/{kwargs.exp_name}/{kwargs.data}_{kwargs.dataset.num_classes}_classes_{kwargs.dataset.num_features}_features/"
        if not kwargs.extract_embeddings:
            if kwargs.models.lambda_kl == 0:
                f"checkpoints/reference_kernel/{kwargs.data}_{kwargs.dataset.num_classes}_classes_{kwargs.dataset.num_features}_features/"  
            if kwargs.models.kernel_only:
                path = f"checkpoints/kernel_only/{kwargs.data}_{kwargs.dataset.num_classes}_classes_{kwargs.dataset.num_features}_features/"          
        os.makedirs(path, exist_ok=True) 
                
        filename = f"localnet_old_seed-{seed}_ep-{total_epochs}_{model_class}"
        if kwargs.data == 'cifar100':
            filename = f"classifier_seed-{seed}_ep-{total_epochs}"
        if kwargs.models.lambda_kl == 0:
            filename = f"refkernel_seed-{seed}_ep-{total_epochs}_{model_class}"
        if kwargs.models.kernel_only:
            if kwargs.data == 'cifar100':
                filename = f"classifier_seed-{seed}_ep-{total_epochs}"
            else:
                filename = f"localnet_old_seed-{seed}_ep-{total_epochs}_{model_class}" #f"kernelonly_seed-{seed}_ep-{total_epochs}_{model_class}"
        
        result_path = f"results/{kwargs.exp_name}/{kwargs.data}_{kwargs.dataset.num_classes}_classes_{kwargs.dataset.num_features}_features"
        if kwargs.models.lambda_kl == 0:
            result_path = f"results/reference_kernel/{kwargs.data}_{kwargs.dataset.num_classes}_classes_{kwargs.dataset.num_features}_features"
        if kwargs.models.kernel_only:
            result_path = f"results/kernel_only/{kwargs.data}_{kwargs.dataset.num_classes}_classes_{kwargs.dataset.num_features}_features"
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
            # Params: {'lambda_kl': 1.191, 'alpha1': 1.001, 'log_var_initializer': 0.2715}
        if kwargs.calibrator_version == 'v2':
            pl_model = AuxTrainerV2(kwargs.models, num_classes=kwargs.dataset.num_classes, 
                                    feature_dim=kwargs.dataset.feature_dim, similarity_dim=kwargs.similarity_dim)              
        else:
            pl_model = AuxTrainer(kwargs.models, num_classes=kwargs.dataset.num_classes)    
        
        if kwargs.corruption_type:
            raw_results_path_test_cal = "results/{}/{}_{}_classes_{}_features/raw_results_test_cal_corrupt_{}_seed-{}_ep-{}_{}.csv".format(
                    kwargs.exp_name,
                    kwargs.data,
                    kwargs.dataset.num_classes,
                    kwargs.dataset.num_features,
                    corruption_text, #kwargs.corruption_type,
                    seed,
                    total_epochs,
                    model_class           
                )
            raw_results_path_train_cal = "results/{}/{}_{}_classes_{}_features/raw_results_train_cal_corrupt_{}_seed-{}_ep-{}_{}.csv".format(
                kwargs.exp_name,
                kwargs.data,
                kwargs.dataset.num_classes,
                kwargs.dataset.num_features,
                corruption_text, #kwargs.corruption_type,
                seed,
                total_epochs,
                model_class                       
            )
            if kwargs.models.lambda_kl == 0:
                raw_results_path_test_cal = "results/{}/{}_{}_classes_{}_features/raw_results_test_cal_corrupt_{}_seed-{}_ep-{}_{}.csv".format(
                    'reference_kernel',
                    kwargs.data,
                    kwargs.dataset.num_classes,
                    kwargs.dataset.num_features,
                    corruption_text, #kwargs.corruption_type,
                    seed,
                    total_epochs,
                    model_class           
                )
                raw_results_path_train_cal = "results/{}/{}_{}_classes_{}_features/raw_results_train_cal_corrupt_{}_seed-{}_ep-{}_{}.csv".format(
                    'reference_kernel',
                    kwargs.data,
                    kwargs.dataset.num_classes,
                    kwargs.dataset.num_features,
                    corruption_text, #kwargs.corruption_type,
                    seed,
                    total_epochs,  
                    model_class                     
                )
            if kwargs.models.kernel_only:
                raw_results_path_test_cal = "results/{}/{}_{}_classes_{}_features/raw_results_test_cal_corrupt_{}_seed-{}_ep-{}_{}.csv".format(
                    'kernel_only',
                    kwargs.data,
                    kwargs.dataset.num_classes,
                    kwargs.dataset.num_features,
                    corruption_text, #kwargs.corruption_type,
                    seed,
                    total_epochs,
                    model_class           
                )
                raw_results_path_train_cal = "results/{}/{}_{}_classes_{}_features/raw_results_train_cal_corrupt_{}_seed-{}_ep-{}_{}.csv".format(
                    'kernel_only',
                    kwargs.data,
                    kwargs.dataset.num_classes,
                    kwargs.dataset.num_features,
                    corruption_text, #kwargs.corruption_type,
                    seed,
                    total_epochs,                       
                    model_class
                )            
        else:
            raw_results_path_test_cal = "results/{}/{}_{}_classes_{}_features/raw_results_test_cal_seed-{}_ep-{}_{}.csv".format(
                    kwargs.exp_name,
                    kwargs.data,
                    kwargs.dataset.num_classes,
                    kwargs.dataset.num_features,
                    seed,
                    total_epochs,
                    model_class           
                )
            if kwargs.data == 'weather' and kwargs.dataset.shift:
                raw_results_path_test_cal = "results/{}/{}_{}_classes_{}_features/raw_results_test_cal_shift_seed-{}_ep-{}_{}.csv".format(
                    kwargs.exp_name,
                    kwargs.data,
                    kwargs.dataset.num_classes,
                    kwargs.dataset.num_features,
                    seed,
                    total_epochs,
                    model_class           
                )
            raw_results_path_train_cal = "results/{}/{}_{}_classes_{}_features/raw_results_train_cal_seed-{}_ep-{}_{}.csv".format(
                kwargs.exp_name,
                kwargs.data,
                kwargs.dataset.num_classes,
                kwargs.dataset.num_features,
                seed,
                total_epochs,    
                model_class                   
            )
            if kwargs.models.lambda_kl == 0:
                raw_results_path_test_cal = "results/{}/{}_{}_classes_{}_features/raw_results_test_cal_seed-{}_ep-{}_{}.csv".format(
                    'reference_kernel',
                    kwargs.data,
                    kwargs.dataset.num_classes,
                    kwargs.dataset.num_features,
                    seed,
                    total_epochs,
                    model_class           
                )
                if kwargs.data == 'weather' and kwargs.dataset.shift:
                    raw_results_path_test_cal = "results/{}/{}_{}_classes_{}_features/raw_results_test_cal_shift_seed-{}_ep-{}_{}.csv".format(
                        'reference_kernel',
                        kwargs.data,
                        kwargs.dataset.num_classes,
                        kwargs.dataset.num_features,
                        seed,
                        total_epochs,
                        model_class           
                    )
                raw_results_path_train_cal = "results/{}/{}_{}_classes_{}_features/raw_results_train_cal_seed-{}_ep-{}_{}.csv".format(
                    'reference_kernel',
                    kwargs.data,
                    kwargs.dataset.num_classes,
                    kwargs.dataset.num_features,
                    seed,
                    total_epochs,                       
                    model_class
                )
            if kwargs.models.kernel_only:
                raw_results_path_test_cal = "results/{}/{}_{}_classes_{}_features/raw_results_test_cal_seed-{}_ep-{}_{}.csv".format(
                    'kernel_only',
                    kwargs.data,
                    kwargs.dataset.num_classes,
                    kwargs.dataset.num_features,
                    seed,
                    total_epochs,
                    model_class           
                )
                if kwargs.data == 'weather' and kwargs.dataset.shift:
                    raw_results_path_test_cal = "results/{}/{}_{}_classes_{}_features/raw_results_test_cal_shift_seed-{}_ep-{}_{}.csv".format(
                        'kernel_only',
                        kwargs.data,
                        kwargs.dataset.num_classes,
                        kwargs.dataset.num_features,
                        seed,
                        total_epochs,
                        model_class           
                    )
                raw_results_path_train_cal = "results/{}/{}_{}_classes_{}_features/raw_results_train_cal_seed-{}_ep-{}_{}.csv".format(
                    'kernel_only',
                    kwargs.data,
                    kwargs.dataset.num_classes,
                    kwargs.dataset.num_features,
                    seed,
                    total_epochs,                       
                    model_class
                )
    
    if kwargs.extract_embeddings: #(kwargs.corruption_type) or              
        best_model_path = path + filename + '.ckpt' #f"localnet_old_seed-{seed}_ep-{total_epochs}.ckpt" # f"classifier_seed-{seed}_ep-{total_epochs}"    
        checkpoint = torch.load(best_model_path, map_location='cpu')
        # print(checkpoint['state_dict'].keys())
        if kwargs.models.lambda_kl == 0:
            pl_model.load_state_dict(checkpoint['state_dict'])
        else:   #strict=False to allow loading only the local net weights into the reference kernel model
            if (kwargs.data in ['tissue', 'cifar100', 'weather']) or (seed == 43):
                pl_model.load_state_dict(checkpoint['state_dict'])
            else:                
                pl_model.load_state_dict(checkpoint['state_dict'])   #        
        print(F'LOADING CHECKPOINT FILE {best_model_path}')
    else:
        print(F'BEGIN CALIBRATION FOR {total_epochs} EPOCHS WITH SEED {seed}!')  
        if kwargs.models.lambda_kl == 0:
            criterion = 'val_con_loss' # 'val_total' 
        else:
            criterion = 'val_kl'
        trainer = pl.Trainer(
                max_epochs=total_epochs,
                accelerator="cuda",
                devices=[cuda_device],
                logger=wandb_logger,
                check_val_every_n_epoch=1,
                #gradient_clip_val=5,
                deterministic=False,
                callbacks=[ #CalibrationPlotCallback(kwargs, dataset.data_train_cal_loader, every_n_epochs=5, device="cuda", type='train'), 
                            #CalibrationPlotCallback(kwargs, dataset.data_test_cal_loader, every_n_epochs=5, device="cuda", type='test'),
                            EarlyStopping(monitor=criterion, #val_total
                                        patience=10, #5
                                        mode="min", 
                                        verbose=True, 
                                        min_delta=0.0),
                            ModelCheckpoint(monitor=criterion, #val_total                                                                                               # Metric to track
                                mode="min",                                                                                                     # Lower is better
                                save_top_k=1,                                                                                                   # Only keep the best model
                                filename=filename, #f"classifier_seed-{seed}_ep-{total_epochs}",                                                           # Static filename (no epoch suffix)
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
        
        path_model = join(path, f"localnet_old_seed-{seed}_ep-{total_epochs}.ckpt") #f"classifier_seed-{seed}_ep-{total_epochs}")
        torch.save(pl_model.model.state_dict(), path_model)
        best_model_path = trainer.checkpoint_callback.best_model_path    
        
        print(F'LOADING CHECKPOINT FILE {best_model_path}')
        checkpoint = torch.load(best_model_path)
        pl_model.load_state_dict(checkpoint['state_dict'])
        
    # path_model = join(path, f"classifier_seed-{seed}_ep-{total_epochs}")
    # torch.save(pl_model.model.state_dict(), path_model)

    if kwargs.models.lambda_kl == 0 or kwargs.models.kernel_only:
        pl_model.build_calibration_memory(dataset.data_val_cal_loader) #data_train_cal_loader
        #kwargs.gamma = pl_model.bandwidth_scalar
    
    raws = []
    pl_model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pl_model.to(device)

    with torch.no_grad():
        for batch in tqdm(dataset.data_train_cal_loader, desc="Extracting pca features"): # tqdm(dataset.data_train_cal_loader, desc="Extracting pca features"):
            if kwargs.dataset.batch_size > 256:
                # Process in smaller mini-batches to avoid OOM
                mini_batch_size = 256
                num_samples = batch[0].size(0)
                for start_idx in range(0, num_samples, mini_batch_size):
                    end_idx = min(start_idx + mini_batch_size, num_samples)
                    mini_batch = [b[start_idx:end_idx].to(device) for b in batch]
                    raw = pl_model.extract_pca(mini_batch)
                    raws.append(raw)                
            else:    
                batch = [b.to(device) for b in batch]                
                raw = pl_model.extract_pca(batch)
                raws.append(raw)

    #all_raws = torch.cat(all_raws)
    print('pca shape: ', raws[1]['features'].shape)
    res, pca = get_raw_res(raws, features=True, adabw=kwargs.models.adabw, reduced_dim=None)
    
    #raws = trainer.predict(pl_model, dataset.data_train_cal_loader) #dataset.data_train_cal_loader
    #res = get_raw_res(raws)
    res.to_csv(raw_results_path_train_cal, index=False)
    
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
                    raw = pl_model.extract_pca(mini_batch)
                    raws.append(raw)                
            else:    
                batch = [b.to(device) for b in batch]                
                raw = pl_model.extract_pca(batch)
                raws.append(raw)

    #all_raws = torch.cat(all_raws)
    print('pca shape: ', raws[1]['features'].shape)
    res, pca = get_raw_res(raws, features=True, adabw=kwargs.models.adabw, reduced_dim=None)
    
    #raws = trainer.predict(pl_model, dataset.data_test_cal_loader)
    #res = get_raw_res(raws)
    res.to_csv(raw_results_path_test_cal, index=False)
    print(raw_results_path_test_cal)
    df = pd.read_csv(raw_results_path_test_cal)
    print(df.shape)

    print("CALIBRATION OVER!")
    
    # Save hp to JSON file
    path = f"hyperparams/{kwargs.data}_{kwargs.checkpoint.num_classes}_classes_{kwargs.checkpoint.num_features}_features"
    os.makedirs(path, exist_ok=True)    
    path_hp = join(path, f"classifier_seed-{seed}_ep-{total_epochs}.json")        
    hp_dict = OmegaConf.to_container(kwargs.models, resolve=True)    
    with open(path_hp, "w") as f:
        json.dump(hp_dict, f, indent=4)

    print("\nSTART TESTING!")     
    
    # latents_list = []
    # with torch.no_grad():
    #     for batch in dataset.data_test_cal_loader:
    #         init_feats, init_logits, init_pca, y_one_hot, init_preds, init_preds_one_hot = batch
            
    #         # move everything you need to the model device
    #         init_feats = init_feats.to(pl_model.device_name)
    #         init_logits = init_logits.to(pl_model.device_name)
    #         init_pca   = init_pca.to(pl_model.device_name)
    #         y_one_hot  = y_one_hot.to(pl_model.device_name)
    #         init_preds = init_preds.to(pl_model.device_name)
    #         init_preds_one_hot = init_preds_one_hot.to(pl_model.device_name)
                                    
    #         noisy_feats = init_feats #+ self.noise * eps

    #         # Forward pass
    #         _, latents_sim = pl_model(noisy_feats, init_logits, init_pca)

    #         # Similarity features (z_cal)
    #         z_cal = latents_sim[:, :pl_model.similarity_dim]

    #         latents_list.append(z_cal)            

    # # Concatenate and ensure they live on the same device as everything else
    # z_cal_full = torch.cat(latents_list, dim=0).to(pl_model.device_name)  # (N_cal, D)    
    
    # # === NEW: estimate bandwidth from z_cal_full ===
    # h = estimate_bandwidth_silverman(z_cal_full)  # (D,)
    
    # # store either per-dim or a global scalar    
    # bandwidth_scalar = float(h.mean().item()) 
    # kwargs.gamma = bandwidth_scalar
    # print('ESTIMATED BANDIWDTH FOR LOCAL METRICS: ', bandwidth_scalar)
           
    test(kwargs)
        

    # optuna: total=0.9974, kl=0.05042, const=0.93656
    # me: total=4.73172, kl=0.05753, const=0.93483
    
    
    