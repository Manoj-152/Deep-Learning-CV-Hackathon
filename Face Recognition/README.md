# Face Recognition using Deep Learning

## Requirements (Libraries)

* torch 1.7.0 
* torchvision 0.8.0
* opencv 3.2.0
* numpy 1.18.5
* matplotlib 3.3.2
* PIL 7.2.0
* tqdm 4.54.0

## Files

* Run the train.py file to train the model.
  - 500 epochs
  - Estimated time for completion: 2 hours
* Run the analysis.py file to find match or mis-match between faces.
  - Accepts paths of two pictures as input using argparse. (reference photo and selfie photo)
  - Example of running the code: python3 analysis.py path_1 path_2
* Run the evaluate.py code to find the match and mis-match accuracies on any other external dataset.
  - Accepts path of the dataset on which the evaluation must be done.
  - Example of running the code: python3 evaluate.py trainset
* Threshold_experimentation.py: For estimating the similarity threshold value.
* Resnet.py: Contains the resnet model class; required to build the resnet18 model.
* Dataloader.py: Contains the FaceDataset class; required to build the trainloader and valloader.
  
## Accuracies on final model (best.ckpt) 
Accuracy for predicting match between faces : 84 %

Accuracy for predicting mis-match between faces : 81 %

(Calculated by using evaluate.py on the validation set)
  
## A Description of the code

* The ResNet18 architecture is used here to generate feature vectors for the faces. After generating the feature vectors, cosine similarity measure is used to find match and mis-match between images.
* The cosine similarity measure is printed here as the confidence score of matching between two face pictures. A threshold of 0.54 was set on the similarity measure for matching and mis-matching of faces.
* The threshold of 0.54 was chosen by analysing using the threshold_experimentation.py file, which could also be run to get a match accuracy vs threshold value and mis-match accuracy vs threshold value graph on the validation set. The graph results are also as follows:
