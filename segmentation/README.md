# Introduction
This repo contains codes for Federated Learning for microvasculature segmentation in OCT-A images. This will be extended towards diabetic retinopathy classification in OCT-A images as well. Currently, the solution uses SFU Vault (OwnCloud) as a drop-off folder to exchange files between the central aggregator server and each client.
## Setup
Please install as per requirements.txt.
## Aggregator Script

The aggregator can be run with the following syntax:

    python fed_aggregator_oc.py -f <client config ini> -c <local config ini>
    
The script uses two ini files, which will be outlined here:

Client configuration (can be named anything). An unlimited number of clients can be set manually, with each client occupying an individual section in the INI file, as shown below.

    [client_name1]
    method = owncloud
    link = (owncloud link)
    pw = (owncloud password)
    [client_name2]
    method = owncloud
    link = (owncloud link)
    pw = (owncloud password)
    
Local configuration (can be named anything). This file contains path information and basic configuration for training.

    [train_params]
    base_dir = /path/to/save/models/
    test_dir = /path/to/load/data/for/testing
    epochs = (int - number of epochs)
    test_tolerance = 0.1 
    
    [resume_params]
    resume_flag = 0 ; 1 if training stalled somewhere and resuming is needed
    resume_epoch = 999  ; 1 + the last successfully-trained epoch
    resume_filestamp = nothing

## Client Training
To minimize the amount of work client-side, the client script is designed to run for a specified wall time without any need for intervention. While running, it checks the drop-off folder regularly and initiates training only when new files are available. The script can be run with the following syntax:

    python train_client_oc.py -f <local config ini> -w <walltime in hours>

The script uses two ini files, which will be outlined here:

Local configuration (can be named anything). This is filled in by the client and is not seen by the aggregator.

    [training]
    train_dir = (folder where training data is located)
	model_dir = (folder where model will be saved)
	batch_size = (batch size used for training)
	[cloud]
	link = 	(shared SFU Vault folder link - will be sent in advance by aggregator)
	pw = (password to above folder)

Training specifications (train_specs.ini). This is sent through the drop-off folder by the aggregator.

    [env]
	name = (unique model name used for this training cycle)
	scratch = (1 if training from scratch, 0 if warm-starting from pre-existing)
	[params]
	current_epoch = (n-th epoch of the training cycle)
	learn_rate = (learning rate for current epoch)

 ### Data File Structure
 The training, validation, and test data should follow this structure (also outlined in img_load_ops.py)
 - train (naming optional)
	 - images
		 - ONLY images
	 - labels
		 - ONLY binarized labels (with same filenames, minus extensions. For example an image named img1.tif would have a label named img1.png (or whatever extension is preferred))
 - test (naming optional)
	 - same as above
 - validation (naming optional)
	 - same as above



