# Facial-Expression-based-Music-Player


# Dependencies
sklearn
Tensorflow
Keras
Pillow
OpenCV

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
  
3. 
