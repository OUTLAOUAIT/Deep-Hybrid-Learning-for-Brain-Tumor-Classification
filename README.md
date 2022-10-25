<Deep-Hybrid-Learning-for-Brain-Tumor-Classification>

## Description

In this repository, we will train a deep learning model for brain tumor recognition.

Using deep learning and machine learning has helped to solve many medical issues. Deep learning based computer vision systems have been especially helpful. Such deep learning systems/models can easily recognize different diseases in medecin when we train them on the right dataset. For example, we can train a deep learning model to recognize different types of diseases in human brain.

Most of the time, obtaining large and well-defined datasets to solve such problems becomes an issue. Because for deep learning models to recognize diseases in human body, they will need to train on huge amounts of data. But that does not mean we cannot train a simple deep learning model to check whether such models can work or not.

In fact, in this work, we will use around 3064 images for training a deep learning model. For brain disease recognition, this may not seem much. But as we will see later on, this is a good starting point, which gives us a deep learning model that works fairly well.

## The Brain Tumor Recognition/Classification Dataset
  
  We will use the [Brain Tumor Dataset](https://figshare.com/articles/dataset/brain_tumor_dataset/1512427) to train a deep learning model in this work.

  This brain tumor dataset contains 3064 T1-weighted contrast-inhanced images with three kinds of brain tumor (glioma, meningioma, pituitary) :
  
  <img src="https://github.com/OUTLAOUAIT/Deep-Hybrid-Learning-for-Brain-Tumor-Classification/blob/main/types.png"/>

   These are mainly three kinds of brain tumors: one is meningioma which contains 708 pictures; the second is glioma which contains 1426 pictures; and lastly there is pituitary tumor which contains 930 pictures. 
  
## Image Pre-Processing
  ### Images normalization
    The brain tumor images have low quality due to noises and low illumination. I proposed to convert the low pixel value images to brighter ones using data normalization and using the min-max normalization function method.

  The MRI image as obtained from the database is unclear. These images also contain a certain amount of uncertainty. Therefore, brain images need to be normalized 
  before further processing. Usually, MRI images look like grey scale images. Hence, the  images are easily normalized to improve the image quality and reduce miscalculation.  This membership function of normalization is mainly used to normalize the image for enhancement with the range 0 to 1. Thus, it is also called the max-min normalization method.

  ###  Data Division and Augmentation
    The deep neural network needs large datasets for better results but our dataset is limited. Our dataset contains 3064 brain images, further divided into 80% for training, which remains for testing and validation purposes. So, data augmentation is needed to change in the minor. I applied rotation, width-shift, height-shift, and the zoom—range for the data requirement. This will enhance the amount of training data, allowing the model to learn more effectively. This may assist in increasing the quantity of relevant data. It contributes to the reduction of overfitting and enhances generalization. 

