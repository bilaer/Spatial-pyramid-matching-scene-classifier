# Spatial Pyramid Matching Scene Classifier
Scene classifier based on k-means algorithm and spatial pyramid matching method

## Introduction
In this implementation, I impleneted a standard k-mean algorithms for generating bag-of-features from training images. And then I use spatial pyramid matching method to generate a vectors of histograms which is used for image classification. I only used 5 classes of images in this implementation and I used 150 keywords for k-mean algorithm. They are .. The overall accuracy is, it is further improvement is possibly by switching the parameters such as the number of centroids. Check for papers in the references for more details.

## Result
A example of the bag-of-features



## Libaries and training data
* [PIL]()
* [Numpy]()
* [pythonCV]() is my own implementation of some computer vision algorithms, which I use to do gaussian smoothing and convolution
* [SUN]() I use SUN image dataset for training and classification

## References
