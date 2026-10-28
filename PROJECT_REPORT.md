
# Project Report: Cats and Dogs Image Classification Using SVM

## 1. Abstract

This project implements a binary image classifier to distinguish cats
from dogs. It uses Histogram of Oriented Gradients (HOG) for feature
extraction and a Linear Support Vector Machine (SVM) for classification.

The project includes dataset preprocessing, model training, validation,
performance evaluation, single-image prediction, and an interactive
Streamlit application.

## 2. Problem Statement

Identifying whether an image contains a cat or a dog is a common
computer vision task. The objective is to implement a classical
machine-learning solution that learns visual patterns from labeled
images and predicts the class of an unseen image.

## 3. Objectives

- Load and validate the supplied training images.
- Extract useful visual features using HOG.
- Train a Linear SVM classifier.
- Evaluate the model on a held-out validation set.
- Build a user interface for image classification.
- Generate predictions for the supplied competition test images.

## 4. Dataset

The dataset is based on the Dogs vs. Cats Redux Kaggle competition.

https://www.kaggle.com/competitions/dogs-vs-cats-redux-kernels-edition/data

The training images are stored in two folders, `train1` and `train2`.
The test images are provided through `test.zip`, and
`sample_submission.csv` defines the output format.

The target classes are Cat and Dog.

## 5. Tools and Technologies

- Python
- NumPy
- Pillow
- scikit-image
- scikit-learn
- Matplotlib
- Streamlit
- joblib

## 6. Methodology

### Image preprocessing
Each valid image is converted to grayscale and resized to 64 x 64 pixels.

### Feature extraction
Histogram of Oriented Gradients represents local edge directions and
shape information in a numerical feature vector.

### Model training
A Linear Support Vector Machine learns the boundary between the Cat
and Dog classes. The dataset is split into training and validation
subsets using stratified sampling.

### Evaluation
Accuracy, F1 score, classification report, and a confusion matrix
are calculated using the validation subset.

### Application
The trained model is used in a Streamlit application for image upload
and prediction. A command-line script is also available.

## 7. Experimental Results

Complete this section after running `python train_model.py`.

- Number of usable training images: [Insert measured value]
- Number of validation images: [Insert measured value]
- Validation accuracy: [Insert measured value]
- Dog-class F1 score: [Insert measured value]

Refer to `artifacts/metrics.json` and
`artifacts/confusion_matrix.png` for the recorded results.

## 8. Limitations

HOG + Linear SVM is a classical computer vision approach. It may not
capture complex image features as effectively as modern deep-learning
models. Prediction quality also depends on image quality and how
representative the training dataset is.

A validation result does not guarantee the same performance on all
real-world images.

## 9. Future Enhancements

- Compare different image sizes and HOG parameters.
- Experiment with alternative SVM configurations.
- Evaluate a convolutional neural network.
- Compare transfer-learning models such as MobileNet.
- Improve the interface with prediction history and batch uploads.

## 10. Conclusion

The project demonstrates a complete classical image-classification
workflow, from image preprocessing and feature extraction to SVM
training, evaluation, and interactive prediction.

## 11. References

- Kaggle Dogs vs. Cats Redux:
  https://www.kaggle.com/competitions/dogs-vs-cats-redux-kernels-edition/data
- scikit-learn Support Vector Machines:
  https://scikit-learn.org/stable/modules/svm.html
- scikit-image HOG documentation:
  https://scikit-image.org/docs/stable/api/skimage.feature.html
- Streamlit documentation:
  https://docs.streamlit.io/