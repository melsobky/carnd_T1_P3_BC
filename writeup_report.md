#**Behavioral Cloning** 


---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* **writeup_report.md** summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model used Nvidia self driving car network

![alt text](.\cnn-architecture.png)

####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 98). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 20). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 104).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the LeNet I thought this model might be appropriate because it has two convolutional layers that i thought was suffecent to detect the road curve. But after some trials with this Network i decided to switch to Nvidia network as it was designed originally for this purpose.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified added a dropout layer.

Then I reduced number of epochs to 2 as i noticed that the mse on the validation set starts to imcrease after the second epoch.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track (areas with different road boundaries), to improve the driving behavior in these case, I added more training data for one sample area in the normaly driving direction and in the opposite direction.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture.

| Layer         	|     Description	        		| 
|:---------------------:|:---------------------------------------------:| 
| Input         	| 160x320x3  image   				| 
| Convolution 5x5     	| 2x2 stride, VALID padding, depth 24	 	|
| RELU			|						|
| Convolution 5x5	| 2x2 stride, VALID padding, depth 36	 	|
| RELU			|						|
| Convolution 5x5	| 2x2 stride, VALID padding, depth 48	 	|
| RELU			|						|
| Convolution 3x3	| 1x1 stride, VALID padding, depth 64	 	|
| RELU			|						|
| Convolution 3x3	| 1x1 stride, VALID padding, depth 64	 	|
| RELU			|						|
| Flatten		| outputs 1164					|
| Fully connected	| Outputs  100					|
| Dropout		|		0.5				|
| Fully connected	| Outputs  50					|
| Fully connected	| Outputs  10					|
| Fully connected	| Outputs  1					|




####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded three laps on track one using center lane driving. Here is an example image of center lane driving:

![center driving](.\center_2017_11_02_02_16_56_457.jpg)

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recenter itself. These images show what a recovery looks like :

![recovering](.\reovering.gif)

To augment the data set, I also flipped images and angles thinking that this would help reduce the -ve angle bias For example, here is an image that has then been flipped:

![original](.\center_2017_10_23_23_12_39_517.jpg)
![flipped](.\center_2017_10_23_23_12_39_517_flipped.jpg)

Etc ....

After the collection process, I had 36006 of data points. I then preprocessed this data by cropping the image to remove the sky and car hood.

![cropped](.\center_2017_10_23_23_12_39_517_cropped.jpg)


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 2 as evidenced by monitoring the mse. I used an adam optimizer so that manually training the learning rate wasn't necessary.
