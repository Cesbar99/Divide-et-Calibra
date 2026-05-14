# Vector Quantized Calibration

data types: 'cifar10', 'cifar100', 'tissue'

num classes: 10, 100, 8

TO EVALUATE COMPETITORS RUN (SMS, DC, TS, PS, IR):

CUDA_VISIBLE_DEVICES=1 python -u run.py pretrain=False calibrate=False quantize=False replicate=False test=False competition=True exp_name=competition data={one_of_the_possible_data_types} gamma=10 dataset.batch_size=128 return_features=True similarity_dim=50 models.max_iter=1000 checkpoint.num_classes={one_of_the-above_values} checkpoint.epochs=9 models.temp_lr=1e-3 n_bins_calibration_metrics=15

ALERT THIS REQUIRES PRE-TRAINED CLASSIFIER WHICH IS MISSING FROM THE REPO!

To solve this problem deo the following:
1) in the cloned repo, create a new folder named results
2) there keep a folder named pre-train
3) inside keep the raw_results for each model (labels, predictions, logits, features)
4) Have a separate file for the calibration data and a file for the test data

Now the code can access the calibration and test data!

DON'T FORGET TO CHECK THE CONFIG FILE (config_local.yaml) AND CHANGE PATHS ACCORDINGLY!

TO RUN A NEW MODEL OR A NEW DATASET:

1) Create a config file for your dataset (use configs/dataset/cifar10.yaml as a template. Either manually add the correct class priors or if you have a lot of classes override them loading the dataset during the following step)
2) Go to data_sets/dataset.py and write your dataloader there (use cifar10 as a template).
3) Go to algorithms/networks.py and add ther your new architecture or model there
4) Go to algorithms/trainers.py and add the training for your custom architecture
5) Go to config.yaml and edit models_map accordingly.
6) Go to replicate.py and add at the top (line 65 circa) your dataset.
7) If your base-model to to pre-train does not support pytorch lightning, go to actions/pretrain.py and add your training code ther 

REMEMBER THAT IF YOUR MODEL DOES NOT HAVE FEATURE REPRESENTATIONS, LOCAL METRICS CANNOT BE COMPUTED!
