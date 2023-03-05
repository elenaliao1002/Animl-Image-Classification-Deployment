# Animl

## Animl Background

Islands around the world are known for their unique native species that cannot be found anywhere else. Unfortunately, the introduction of non-native invasive species by visitors to these islands poses a significant threat to these endemic species. To safeguard these species, The Nature Conservancy is employing various strategies, including the use of wireless camera traps and machine learning to detect invasive animal incursions in real time. The challenge is to accurately flag images that contain potential invaders while ensuring that the native species are not misidentified.

## Animl and Image Classification

An object detection algorithm and species classification model have been implemented in Animl, an image review platform, but further tuning is necessary to differentiate between native and non-native species. The Nature Conservancy develop and document a pipeline for exporting labeled, human-validated images and annotations from Animl for future model training. And also retrain image classification models specifically for Santa Cruz Island and deploy the model within the existing Animl AWS infrastructure. By employing machine learning and data science, The Nature Conservancy hopes to provide real-time monitoring to various native species, including the Santa Cruz Island Fox, and mitigate the risk of invasive species.

## Animl Classifer Training Structure

This README describes how to create a classifier for animal species using images from Santa Cruz island.

1. Retrieve the COCO .json file from Animl, which comprises image metadata such as filename and location, along with annotations containing label data. Please note that this JSON file does not contain the actual image files, but rather the associated metadata.
2. Download all the full size images from AWS S3 referenced in the cct.json(COCO) file.
3. Create a classification label specification JSON file (same format that MegaDetector outputs).
4. Implement MegaDetector to locate animals in the images and crop them out.
5. Create classification dataset by spliting the images into 3 sets (train, val, and test).
6. Start to train a classifer.

## Preparation

#### Clone Relative GitHub Repo and Set Up Environment

To create an environment, we need to clone the following repo and install prerequisites(Anaconda, Git, aws-vault).

* [CameraTraps](https://github.com/Microsoft/cameratraps) repo
* [microsoft/ai4eutils](https://github.com/microsoft/ai4eutils) repo
* [animl-analytics](https://github.com/tnc-ca-geo/animl-analytics) repo
* [animl-ml](https://github.com/tnc-ca-geo/animl-ml) repo

##### Code

```bash
### install anaconda 
wget https://repo.anaconda.com/archive/Anaconda3-2022.10-Linux-x86_64.sh
bash Anaconda3-2022.10-Linux-x86_64.sh
source .bashrc

### install brew and git
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
brew install git 

### git clone relative repo
git clone https://github.com/Microsoft/cameratraps CameraTraps
git clone https://github.com/microsoft/ai4eutils ai4eutils
git clone https://github.com/tnc-ca-geo/animl-analytics animl-analytics
git clone https://github.com/tnc-ca-geo/animl-ml animl-ml

### create environment
conda create -n cameratraps-classifier
conda env update -f ~/CameraTraps/environment-classifier.yml --prune

### verifying that CUDA is available (and dealing with the case where it isn't) --parallel computing platform 
python ~/CameraTraps/sandbox/torch_test.py

"""If CUDA isn't available it would return : `CUDA available: False` and please do the following step"""
pip uninstall torch torchvision
conda install pytorch=1.10.1 torchvision=0.11.2 -c pytorch

### install aws-vault 
brew install aws-vault

### Optional steps to make classification faster in Linux
conda install -c conda-forge accimage
pip uninstall -y pillow
pip install pillow-simd

### setting environment variables in `.bashrc`
# Python development
export PYTHONPATH="/path/to/repos/CameraTraps:/path/to/repos/ai4eutils"
export MYPYPATH=$PYTHONPATH
export BASE_LOGDIR="/home/<user>/CameraTraps/classification/BASE_LOGDIR"

# accessing MegaDB
export COSMOS_ENDPOINT="[INTERNAL_USE]"
export COSMOS_KEY="[INTERNAL_USE]"

# running Batch API
export BATCH_DETECTION_API_URL="http://[INTERNAL_USE]/v3/camera-trap/detection-batch"
export CLASSIFICATION_BLOB_STORAGE_ACCOUNT="[INTERNAL_USE]"
export CLASSIFICATION_BLOB_CONTAINER="classifier-training"
export CLASSIFICATION_BLOB_CONTAINER_WRITE_SAS="[INTERNAL_USE]"
export DETECTION_API_CALLER="[INTERNAL_USE]"
```

#### AWS Configuration

To download the full size images from AWS S3 storage infrastructure, we need to do the configuration:

```bash
aws configure
"""
AWS Access Key ID [None]: enter your Key ID
AWS Secret Access Key [None]: enter your Access Key
Default region name [None]: us-west-2
Default output format [None]: json
"""

### verify configuration settings to access credentials in files
aws s3 ls s3://animl-images-archive-prod

```

#### Build Folder Structure

Add additional directories (`~/classifier-training`, `~/images`, `~/crops`, etc.) so that the contents of your `home/` directory matches the following structure:

```
├── ai4eutils/                  # Microsoft's AI for Earth Utils repo
├── animl-analytics/            # animl-analytics repo (utilities for exporting images)
├── animl-ml/                   # This repo, contains Animl-specific utilities
├── CameraTraps/                # Microsoft's CameraTraps repo
│   ├── classification/
│   │   ├── BASE_LOGDIR/        # classification dataset and splits
│   │   │   └── LOGDIR/         # logs and checkpoints from a single training run
│   ├── classifier-training/
│   │   ├── mdcache/            # cached "MegaDetector" outputs
│   │   │   └── v5.0b/          # NOTE: MegaDetector is in quotes because we're
│   │   │       └── datasetX.json # also storing Animl annotations here too
│   │   └── megaclassifier/     # files relevant to MegaClassifier
├── crops/                      # local directory to save cropped images
│   └── datasetX/               # images are organized by dataset
│       └── img0___crop00.jpg
└── images/                     # local directory to save full-size images
    └── datasetX/               # images are organized by dataset
        └── img0.jpg

```

## Training Pipeline

#### Step1 : Exporting labeled, human-validated images and annotations from Animl

In the Animl Interfece, the following filters make sense when you're exporting the data:

- fox
- bird
- skunk
- rodent
- lizard

Then download the data with selected labels from the Animl interface by clicking EXPORT TO COCO format. Then we get the cct.json file. We can use it to download the images later on.

#### Step2 : Download all the full size images referenced in the cct.json(COCO) file

This code downloads image files from Amazon S3 and saves them to a local directory.

The code takes two arguments: "--coco-file", the path to the coco file, and "--output-dir", the local directory to download the images to.

```bash
python ~/animl-analytics/utils/download_images.py \
 --coco-file  ~/classifier-training/mdcache/v5.0b/<dataset_name>_cct.json\
 --output-dir ~/images/<dataset_name>

"""remember to change the dataset_name"""
```

#### Step3 : Create a classification label specification JSON file(same format that MegaDetector outputs)

Create a classification label specification JSON file (usually named label_spec.json). This file defines the labels that our classifier will be trained to distinguish, as well as the original dataset labels and/or biological taxa that will map to each classification label.

Some of the following steps expect the image annotations to be in the same format that MegaDetector outputs after processing a batch of images. To convert the COCO for Cameratraps file that we exported from Animl to a MegaDetector results file, navigate to the /home/studio-lab-user/ directory and run:

```bash
python animl-ml/classification/utils/cct_to_md.py \
  --input_filename ~/classifier-training/mdcache/v5.0b/<dataset_name>_cct.json \
  --output_filename ~/classifier-training/mdcache/v5.0b/<dataset_name>_md.json
"""remember to change the dataset_name"""
```

## Training Pipeline

Result
------

```

```
