#**Traffic Sign Recognition** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. 

You're reading it! and here is a link to my [project code]https://github.com/grafandreas/CarND-Traffic-Sign-Classifier-Project-Sol)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32x32x1
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing the number of signs per class and
the ratio of training/validation/and test set.

[image1]: https://github.com/grafandreas/CarND-Traffic-Sign-Classifier-Project-Sol/blob/master/writeup/statistics.png "x"

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data.

As a first step, I decided to convert the images to grayscale because it is good practice to reduce the number
of features only to the relevant ones. Traffic signs are designed to be recognizable by shape only, color is not
required, so performance is improved and dependency on lighting conditions is reduced.

However, the images vary greatly in brightness - some are very bright, other are almost all black. To normalize images,
I used the normalize functionality of OpenCV. I also experimented with equalizing the Histogram, which results in 
a slightly different grey distribution, but could not identify significant changes in the accuracy of the neural net.

In addition, I normalized the pixel values from 0..256 to -1..1

However, this still resulted in overfitting. So I decided to generate fake data. This was done by creating two additional images per image,
slightly rotated both clock- and counter-clock-wise. The rotation is 3 degree, which was a first guess, but resulted immediately in a better performance. It was assumed, that larger rotation values would be irrelevant, since most traffic signs should be almost upright. Additional transformation could have been slight perspective transformations et. al, but the transformation was sufficient.


Here is an example of the final image, the left image is normalized and to see if there is a difference, the right image is equalized.

[image2]: ./writeup/normalize_and_equalize.png "y"

I did not include images of the augmented data, because the rotation is hardly visible.



####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the basic LeNet architecture, augmented with a few drop outs. Drop outs were added since the initial result with plain LeNet did not yield in satisfactory results. In addition, the first convolution was increased from 6 to 10.


| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 RGB image   							| 
| Convolution       	| Input = 32x32x1. Output = 28x28x10. 	        |
| RELU					|												|
| Max pooling	      	| Input = 28x28x6. Output = 14x14x10.			|
| Convolution  		    | Output = 10x10x16.     						|
| Droupout				| 50%											|
| RELU					|												|
| Max pooling	      	| Input = 10x10x16. Output = 5x5x16.			|
| Flatten				| Input = 5x5x16. Output = 400. 				|
| Fully connected		| Input = 400. Output = 200.        									|
| RELU					|												|
| Dropout				| 5%       								|
| Fully connected		|   Input = 200. Output = 120.     									|
| RELU					|												|
| Fully connected		|  Input = 120. Output = Number of labels   									|
|						|												|
 


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

EPOCHS: I started with 10 epochs. During the tuning, it was increased to 15, but it turned out that the correct dropout was much more efficient, so I could reduce it back to 10
BATCH SIZE: 128, kept the original values
Learning rate: 0.001

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The idea was to try to expand on an existing architecture, since the recommendation is usually to start from something simple. Since LeNet was used, this was the initial idea. Alternatives could have been Alexnet and others.  Some experimentation seemd to indicate that drop-out did not give much improvement, until I noticed, that dropout was activated both for training and classification. After refactoring the code, it turned out that drop-out was a significant improvement and I rolled back my changes to Epochs and batch sizes.

My final model results were:
* validation set accuracy of > 0.97, depending on the test run.
* test set accuracy of 0.94


 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

I took the liberty to take some photos from real german traffic signs, since I wanted to experiment on angles etc.:

[image4]./custom_sings/keepr1_38.jpg ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Keep Right     		|Keep Right 								| 
| 50 km/h    			| 120 km/h										|
| 50 km/h					| 50 km/h											|
| Priority Road   		| Priority Road				 				|
| Yield		| Yield      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This is lower than the accuracy of the test set, but the number of examples is too small for relevant statistics.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabiliti

For the second  image, the model is relatively sure that this is a stop sign (probability of 1), and the image does contain50 km/h limit. The top five soft max probabilities were

| Probability			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1    		|Keep Right 								| 
| 1    			| 120 km/h										|
| 1				| 50 km/h											|
| 1  		| Priority Road				 				|
| 1	| Yield      							|


