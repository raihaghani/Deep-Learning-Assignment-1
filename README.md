## EmoMTL-CNN: Multitask Facial Emotion Recognition with VGG16 and ResNet50
A deep learning project implementing and comparing multitask CNN models for facial emotion analysis. The system jointly performs facial expression classification and continuous valence–arousal regression using pretrained VGG16 and ResNet50 backbones.

## Project Overview:
This project explores multitask learning for emotion recognition from facial images. Instead of training separate models for each emotional attribute, a single neural network predicts three outputs simultaneously:
Facial expression classification across 8 emotion categories
Valence regression, representing emotional positivity or negativity
Arousal regression, representing emotional intensity
Two ImageNet-pretrained CNN backbones, VGG16 and ResNet50, were evaluated and fine-tuned using a shared feature extractor with task-specific output heads.

## Dataset:
The dataset contains 3,999 facial images with annotations for expression class, valence, and arousal.

Expression labels: 8 classes
Valence range: −1 to +1
Arousal range: −1 to +1
Training set: 2,799 images
Validation set: 600 images
Test set: 600 images

The dataset was approximately balanced across expression classes, with around 500 samples per class.

## Methodology:
Images were resized to 224×224, normalized, and augmented using random flips, rotations, and contrast adjustments. Both VGG16 and ResNet50 were used as pretrained feature extractors, followed by three task-specific heads:

Expression head: Dense layers with dropout and softmax activation
Valence head: Dense layers with dropout and tanh activation
Arousal head: Dense layers with dropout and tanh activation
The models were trained using the Adam optimizer with categorical cross-entropy for expression classification and mean squared error for valence and arousal regression.

## Experiments and Results:
The VGG16 baseline achieved low expression classification accuracy, around 12–14%, while performing better on the regression tasks with valence MAE around 0.38 and arousal MAE around 0.32.
The ResNet50 model showed more stable convergence and stronger regression performance compared to VGG16. Expression classification remained challenging, reaching approximately 15% accuracy, but ResNet50 demonstrated better fine-tuning potential due to its deeper residual architecture.

## Key Findings:
Multitask learning was successfully implemented for joint emotion classification and regression.
Valence and arousal prediction performed better than expression classification.
ResNet50 outperformed VGG16 in convergence stability and regression performance.
Expression classification was limited by the relatively small dataset size of around 500 samples per class.
Residual learning in ResNet50 provided better feature reuse and fine-tuning potential.

## Technologies Used:
Python
TensorFlow / Keras
VGG16
ResNet50
Transfer Learning
Multitask Learning
Computer Vision
Emotion Recognition

## Conclusion:
This project demonstrates the use of multitask deep learning for facial emotion recognition by combining categorical expression classification with continuous valence–arousal regression. ResNet50 showed better overall performance than VGG16, especially for regression tasks, while expression classification remained a challenging problem due to limited dataset size.
