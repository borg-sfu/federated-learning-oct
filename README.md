# Introduction
This repo contains codes for Federated Learning for diabetic retinopathy classification in OCT-A images. Currently, the solution uses SFU Vault (OwnCloud) as a drop-off folder to exchange files between the central aggregator server and each client, with eventual expansion to a separate computer.

## Setup
 The relevant dependencies can be installed as follows (or through pip install -r requirements_classification.txt):

### Install with conda or pip

    conda create --name fedoc python=3.7.4
    activate fedoc
    conda install tensorflow-gpu==2.1 tensorflow-estimator==2.1 configparser==5.0.0 opencv-python==3.4.4.19 imgaug==0.4.0 requests==2.24.0 pandas==1.1.3 xlrd==1.2.0 more-itertools==8.5.0 scikit-learn==0.23.0 num2words==0.5.10 h5py==2.10.0 
    conda install -c conda-forge tqdm
    
### Next, install pyocclient through its repo:

    git clone https://github.com/owncloud/pyocclient.git
    cd pyocclient
    git checkout e26af99f9c14ed78e5ee494ced271a36368d1095
    python setup.py install
    

# Classification

## Client Training
To minimize the amount of work client-side, the client script is designed to run for a specified wall time without any need for intervention. While running, it checks the drop-off folder regularly and initiates training only when new files are available. The script can be run with the following syntax:

### Run this client training script once first to preprocess the images -> the aggregator code requires one fold generated from this train_client_oc.py script

$ python train_client_oc.py -f <local config ini> -w <walltime in hours> -p <polltime in seconds> -ev <boolean for whether to generate intensity profile of the volume> -ID <institution ID to stamp each saved .npy file with the institution in the scenerio where one institution may split their data into two silos>

The script uses two ini files, which will be outlined here:

### Local configuration (can be named anything). This is filled in by the client and is not seen by the aggregator. A sample local configuration file (local_client_test.ini) is available to be downloaded.

	[training]
	data_dir = (folder where training data is located)
	model_dir = (folder where model will be saved)
	batch_size = (batch size used for training)
	image_size = (image resolution in one dimension)
	weights = (None, imagenet, or path to weights)
	excel_file = (loading data using spreadsheet: input path to spreadsheet (eg. /path/to/spreadsheet/spreadsheet.xlsx; loading data using class encoded file names: leave blank (eg. '')
	aug_factor = (int, all classes will be upsampled through augmentation to this factor multiplied by the most prevalent class)
	original_stratification = (list of original stratification separated by ';' (eg. Normal;Mild;Mod;Severe;PDR)
	class_definitions = (list of original stratification separated by ';' and the split separated by '_' (eg. Normal;Mild_Mod;Severe;PDR)
	classes = (names of classification classes separated by ';' (eg. NRDR;RefDR))
	image_breakdown = (list of image channels separated by ';' (eg. SVC-angio;DVC-angio;MIP-struct))
	model_type = (eg. VGG19, VGG16, or ResNet50)
	num_layers = (number of layers to freeze)
	num_folds = (number of folds to split the data into - one fold will be for testing and one for validation... the rest will be for training)

	[cloud]
	link = 	(shared SFU Vault folder link - will be sent in advance by aggregator)
	pw = (password to above folder)
	

### Training specifications (train_specs.ini). This is sent through the drop-off folder by the aggregator.

	[env]
	name = (unique model name used for this training cycle)
	scratch = (1 if training from scratch, 0 if warm-starting from pre-existing)

	[params]
	current_epoch = (n-th epoch of the training cycle)
	learn_rate = (learning rate for current epoch)
	outer_fold = (int - representing the current outer fold index in nested cross validation)
	inner_fold = (int - representing the current inner fold index in nested cross validation)

### Aggregator local configuration (fed_config.ini) - can rename and referred to through argparse user inputs

	[params]
	image_size = (int) 1D size for example (512) which will result in 512x512x3 square image for training. Edit the img_loading.py code if non-square images are used.
	num_folds = (int) the number of folds for nested cross validation (i.e. 5 will result in 20 trained models (5x outer folds and 4x inner folds)
	institution_ID = institution ID of the hosting client - this directs the aggregator to one fold of data to verify that there are no malicious models 
	base_dir = /path/to/directory/to/save/the/models
	test_dir = /path/to/directory/containing/test/data (should correspond to 'data_dir' on the host client's local config client file
	epochs = 100 (number of epochs to train for)
	max_lr = 0.0003 (maximum learning rate - this is the learning rate used if 'steady' learning rate is selected)
	min_lr = 0.000001 (minimum learning rate - minimum LR set)
	num_lr_cycles = 50 (number of changes to learning rate)
	lr_type = step_decay (learning rate type - steady, step_decay, tri1, tri2) 
	test_tolerance = 0.3 (test tolerance to filter models with accuracy below test_tolerance)
	
#### Defining each learning rate type
steady - constant learning rate (specified by max_lr)
step_decay - decay for num_lr_cycles times from max_lr to min_lr 
tri1 - cyclic learning rate cycling num_lr_cycles times cycling from max_lr to min_lr
tri2 - cyclic learning rate with decay factor of 0.5 (can be changed in function lr_scheduler in fed_aggregator_oc.py)


# Data File Structure (Classification)

### How to import from an excel file:
See 'Data_Summary_Example.xlsx' for reference. If the columns change, corresponding functions in the image loading should also be altered. The format of the excel file should be as follows:
    
import from excel file - the format of the excel file should be as follows (an example of our use case):

    - Label (first column)
        - column of stratified label (eg. Normal, Mild, Mod, etc.. the naming convention should correspond to 'original_stratification' in the Local Configuration file described above)
    - SVC_angio (second column)
        - column of path to the corresponding SVC OCT-A
    - DVC_angio (third column)
        - column of path to the corresponding DVC OCT-A
    - SVC_struct (fourth column)
        - column of path to the corresponding SVC structural OCT
    - DVC_struct (fifth column)
        - column of path to the corresponding DVC structural OCT

