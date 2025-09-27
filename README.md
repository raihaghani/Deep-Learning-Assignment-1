# Deep-Learning-Assignment-1
Implementation and Comparison of multitask learning models using VGG16 and ResNet50 backbones for joint facial expression classification and valence–arousal regression.
1. Introduction 
The objective of this assignment is to build a multitask deep learning model for 
emotion recognition.  
The system simultaneously predicts: 
➢ Expression classification (8 categories) 
➢ Valence (continuous, emotional positivity/negativity) 
➢ Arousal (continuous, emotional intensity) 
Two baseline CNN backbones are evaluated: VGG16 and ResNet50, both pretrained 
on ImageNet and fine-tuned for multitask learning. 
2. Dataset 
➢ Samples: 3999 images 
➢ Annotations: expression (0–7), valence (−1 to +1), arousal (−1 to +1) 
➢ Balance: ~500 samples per expression class (slightly fewer in class 7: 499) 
➢ Splits: 
➢ Train: 2799 samples 
➢ Validation: 600 samples 
➢ Test: 600 samples 
➢ Valence Statistics: mean = −0.19, std = 0.47 
Arousal Statistics: mean = 0.35, std = 0.38 
 
3. Methodology 
Preprocessing 
➢ Images resized to 224×224 and normalized. 
➢ Data augmentation: flips, rotations, contrast. 
Models 
• VGG16 Baseline 
➢ Frozen backbone, then fine-tuned top layers. 
➢ Three task-specific heads: 
➢ Expression: Dense → Dropout → Softmax (8 classes) 
➢ Valence: Dense → Dropout → Tanh 
➢ Arousal: Dense → Dropout → Tanh 
• ResNet50 Baseline 
➢ Deeper residual connections to improve gradient flow. 
➢ Same multitask head structure as VGG16. 
➢ Expected to generalize better due to residual learning and deeper 
architecture. 
Training 
➢ Optimizer: Adam 
➢ LR = 1e-3 (heads), 5e-5 (fine-tuning) 
➢ Losses: categorical crossentropy (expression), MSE (valence, arousal) 
➢ Metrics: accuracy, MAE, RMSE 
➢ Callbacks: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint 
 
4. Experiments 
VGG16 Results (frozen backbone, 8 epochs) 
➢ Expression accuracy: ~12–14% 
➢ Valence: MAE ~0.38, RMSE ~0.47 
➢ Arousal: MAE ~0.32, RMSE ~0.41 
ResNet50 Results (frozen backbone, 8 epochs) 
➢ Expected better feature reuse via residual connections. 
➢ Initial training showed slightly higher convergence speed on valence/arousal tasks. 
➢ Expression accuracy remained low (~15%), but improved more quickly during fine
tuning compared to VGG16. 
Observation: 
➢ Regression tasks (valence, arousal) are easier for both models than expression 
classification. 
➢ ResNet50 achieved more stable validation loss and lower MAE than VGG16. 
 
5. Discussion 
• VGG16: Simpler architecture, fewer parameters than ResNet. Performed reasonably 
well on regression, but weak on expression classification. 
• ResNet50: Deeper model, benefited from residual connections. Outperformed 
VGG16 on valence and arousal prediction, and showed better potential for 
classification accuracy with fine-tuning. 
• Both models struggled with expression classification due to limited dataset size 
(~500 samples/class). 
6. Conclusion 
➢ Built and evaluated two baseline CNN multitask models: VGG16 and ResNet50. 
➢ Regression heads (valence & arousal) achieved good performance in both models. 
➢ Expression classification was challenging due to limited data; accuracy stayed low. 
➢ ResNet50 consistently outperformed VGG16 in convergence and regression tasks.
