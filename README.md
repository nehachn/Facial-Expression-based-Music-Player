# Facial-Expression-based-Music-Player
# Objective
To develop an algorithm to automatically detect mood of a person through his/her facial expressions and play songs corresponding to the detected mood. 

# Introduction
Studies have shown that music has a direct effect on moods of humans. People tend to listen to music based on there mood and interests. The human face acts as the main indicator for the behavioral and the emotional state of the individual. Facial expression can be recognized to detect the mood of a person. It can further be used to make suggestions for music as it is often a tedious job to search from a the huge number of songs present now-a-days on internet. 

We present a model which eliminates the time-consuming and the tedious work ofmanually playing the songs from any playlist available on the Web after detecting the mood of the individual.

# Dependencies
  sklearn
  
  Tensorflow
  
  Keras
  
  Pillow
  
  OpenCV
  
# Training dataset
Training dataset is FER2013, which consists of approx 24 thousand images belonging to 5 different classes namely : Angry, Happy, Neutral, Sad, Surprise.

# Running Code #
1. To download the dependencies run the following code in terminal: 
  ```
    pip install -r requirements.txt
  ```
2. Now test facial expressions using various models:  
  a. Using CNN: run ```python CNN/main.py```  
  b. Using ResNet-50:   
      i. Download trained model from [here](https://drive.google.com/file/d/1KJhlFYyLkwaUypJg35xIFbZS2B0meGDq/view?usp=sharing) in models directory  
      ii. run ```python ResNet50/resnet.py```   
  c. Using VGG16:   
      i. Download trained model from [here](https://drive.google.com/file/d/1S5SvTpzJTCW29hNy2HEn0XWaEQsvo-Nh/view?usp=sharing) in models directory   
      ii. run ```python VGG16/vgg_test.py```    
  d. Using pseudo-labeling: run ```python Pseudo_Labeling/pseudo_test.py```
  
3. Run the ```main.py``` 
   Press spacebar to capture current image and then the song corresponding to the detected mood will open up on the bowser (firefox)
