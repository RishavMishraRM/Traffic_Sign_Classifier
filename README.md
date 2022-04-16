# **Traffic Sign Recognition** 


### **Build a Traffic Sign Recognition Project**
The goals/steps of this project are the following:
<a href="https://raw.githubusercontent.com/neelrast/lenet-traffic-sign-classifier/master/README.md"></a>
  
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy, and/or pandas methods rather than hardcoding results manually.
I used the NumPy library to calculate summary statistics of the traffic
signs data set
* The size of training set is <b>`34799`</b>
* The size of the validation set is <b>`4410`</b>
* The size of test set is <b>`12630`</b>
* The shape of a traffic sign image is <b>`(32, 32, 3)` # 3 becuase of R,G,B Channels.</b>
* The number of unique classes/labels in the data set is <b>`43`</b>


#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data is distributed alongside our 43 unique label points.

<img src="./images/barchart.png">


As we can see labels marked as `Speed limit(30km/h)`, `Speed limit(50km/h)`, `Speed limit(70km/h)` & `Speed limit(80km/h)`, along with `No Passing for vehicles over 3.5 metric tones`,`Priority work`,`Yeild`, `Road Work`, `Keep right` are among the top labels with majority of samples. Whereas, certain labels like `Speed limit(20km/h)`, `Dangerous Curve to the left`, `Pedestrians` and `Go Straight or left` are among the few labels with least amount of samples.

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen, and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.
