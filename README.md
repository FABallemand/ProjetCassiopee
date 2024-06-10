# 🌌 Projet Cassiopée

## 🖐 Self-supervised learning with deep neural networks for automatic gesture recognition
Gesture is an important means of non-verbal communication that facilitates many human interactions in different contexts, including driving or communicating with disabled people. It is therefore essential to acquire the ability to automatically recognise this means of communication in order to improve human-computer interactions.

Current progress in the development of automatic recognition of human gestures has accelerated but is hampered by the costly human annotation required to establish the necessary dataset. One solution to this problem is to use self-supervised learning, which extends the learning process applied to unlabelled data. This methodology has been widely applied to several other computer vision tasks, but its application to gesture recognition, particularly in a multimodal context, is still limited.

## 👥 Team
- ALLEMAND Fabien
- MAZZELLA Alessio
- VILLETTE Jun

![](img/cassiopee_1.png)

## Utilisation

1. Description
- documentation contains the poster and the report of the project
- demonstration contains the necessary to run an interactive hand gesture demonstration
- data contains the datasets used (mocaplab data not available publicly)
- src contains all the code

2. Running
- Train all three supervised models : src/models/mocaplab/full_train.py
- Train self-supervised CNN : src/models/mocaplab/ssl_train.py
- Visualize the classifications of the models : src/models/mocaplab/classification_visu.py
- Plot GradCAM visualization skeletons : src/visualisation/plot_points_color.py

## Method

1. Working on professional high-quality motion capture data provided by Mocaplab

![](img/skeletons.png)

2. Three deep learning models in supervised settings : Fully connected (FC), CNN and LSTM

3. A self-supervised learning approach for CNN

![](img/models_architecture.jpeg)

4. Explainability : GradCAM visualization

![](img/gradcam.png)

## Results

1. Deep learning methods are very powerful for gesture recognition

2. Self-supervised learning leverages small amount of labeled data for better results than supervised learning

3. Limitations and perspectives
- Binary classification is an ‘easy’ task on which simple models can excel 
- Explaining the prediction for two hands signs needs further investigation  
- Extend to multiple categories data and consider larger volumes of data

![](img/results_tab.png)

## Example of interface for demonstration

## Citation
