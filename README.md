# Animl

## Animl Background

Islands around the world are known for their unique native species that cannot be found anywhere else. Unfortunately, the introduction of non-native invasive species by visitors to these islands poses a significant threat to these endemic species. 

## Camera Trap and Machine Learning

To safeguard these species, The Nature Conservancy is employing various strategies, including the use of wireless camera traps and machine learning to detect invasive animal incursions in real time. Currently, a pilot network of wireless wildlife cameras is deployed on Santa Cruz Island, located off the southern California coast. These cameras capture images triggered by movement and transmit them to the cloud. The challenge is to accurately flag images that contain potential invaders while ensuring that the native species are not misidentified. 

## AniML and Image Classification

An object detection algorithm and species classification model have been implemented in Animl, an image review platform, but further tuning is necessary to differentiate between native and non-native species. The Nature Conservancy develop and document a pipeline for exporting labeled, human-validated images and annotations from AniML for future model training. And also retrain image classification models specifically for Santa Cruz Island and deploy the model within the existing AniML AWS infrastructure. By employing machine learning and data science, The Nature Conservancy hopes to provide real-time monitoring to various native species, including the Santa Cruz Island Fox, and mitigate the risk of invasive species.

## Animl Classifer Training Pipeline

This README describes how to create a classifier for animal species using images from Santa Cruz island.

1. Retrieve the COCO .json file from Animl, which comprises image metadata such as filename and location, along with annotations containing label data. Please note that this JSON file does not contain the actual image files, but rather the associated metadata.
2. Download all the full size images referenced in the cct.json(COCO) file.
3. Create a classification label specification JSON file (same format that MegaDetector outputs).
4. Implement MegaDetector to locate animals in the images and crop them out.
5. Create classification dataset by spliting the images into 3 sets (train, val, and test).
6. Start to train a classifer.

## Preparation

#### Clone Relative GitHub Repo and Set Up Environment

-[CameraTraps](https://github.com/Microsoft/cameratraps) repo
-[microsoft/ai4eutils](https://github.com/microsoft/ai4eutils) repo
-[animl-analytics](https://github.com/tnc-ca-geo/animl-analytics) repo
-[animl-ml](https://github.com/tnc-ca-geo/animl-ml) repo

#### Build Folder Structure




Result
------

```

```
