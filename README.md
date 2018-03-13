# Spatial Pyramid Matching Scene Classifier
Scene classifier based on k-means algorithm and spatial pyramid matching method

## Introduction
In this implementation, I impleneted a standard k-mean algorithms for generating bag-of-features from training images. And then I use spatial pyramid matching method to generate a vectors of histograms which is used for image classification. I only used 6 classes of images in this implementation and I used 150 keywords for k-mean algorithm. The overall accuracy is around 0.3, Further improvement is possibly by switching the parameters such as the number of centroids. Check papers in the references section for more details.

## Result
Visual words of training image

![alt text](https://github.com/bilaer/Spatial-pyramid-matching-scene-classifier/blob/master/filter37.jpg)
![alt text](https://github.com/bilaer/Spatial-pyramid-matching-scene-classifier/blob/master/visualWord.jpg)

## Libaries and training data
* [PIL](https://pillow.readthedocs.io/en/latest/) I use PIL to open and translate images into numpy array
* [Numpy](http://www.numpy.org/) I use Numpy to do image convolution and othe scientific calculation.
* [pythonCV](https://github.com/bilaer/PythonCV) is my own implementation of some computer vision algorithms, which I use to do gaussian smoothing and convolution.
* [SUN](http://vision.princeton.edu/projects/2010/SUN/) I use SUN database for training and classification

## References
Lazebnik, Svetlana, Cordelia Schmid, and Jean Ponce. "Beyond bags of features: Spatial pyramid matching for recognizing natural scene categories." Computer vision and pattern recognition, 2006 IEEE computer society conference on. Vol. 2. IEEE, 2006.
