# **Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images/arch.PNG "NvidiaModel"
[image2]: ./images/model_screen.PNG "FinalModel"


---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

The project includes the following files:
* model.py containing the script to create and train the model.
* drive.py for driving the car in autonomous mode (provided by Udacity).
* model.h5 containing a trained convolution neural network. 
* Writeup.md that summarize the results.

#### 2. Submission includes functional code
With the help of Udacity car simulator and my drive.py file (code provided by Udacity), the car can be driven autonomously,based on the model trained,  around the track by executing  
```sh
python drive.py model.h5
```
And running the simulator at the same time.

#### 3. Submission code is usable and readable

The model.py file contains the code for training, saving and reporting the convolution neural network. The file shows the pipeline used for training and validating the model. The code is properly commented. The model used was the Nvidia Selfdriving car. Another solution, proposal, is commented in the code , just to have another architecture option. 

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

According to the recommendation of Udacity, the neural network created by udacity was implemented. This network is detailed in this [Nvidia Model](https://devblogs.nvidia.com/deep-learning-self-driving-cars/).
This specific model have 9 layers, including a normalization layer, 5 convolutional layers, and 3 fully connected layers.

This was the approach that was taken from the beginning to the end of the project, in order to to take advantage of the pre-processing and data augmentation without modifying the proposed model.

In the next figure the model is presented

![NvidiaModel][image1]

My initial approach was to use LeNet, but it was hard to have the car inside the street with three epochs (this model could be found here). After this, I decided to try the nVidia Autonomous Car Group model, and the car drove the complete first track after just three training epochs (this model could be found here).

The normalization layer (lambda layer) is crucial to prepare the best environment to the training of the optimization model (loss function). This would make the training a faster process.

The architecture, meaning: the number of layers, the size of the kernel, the strides, etc. were parameter chosen empirically

The optimization mode use a mean square error (MSE) as a loss function and ADAM optimizer as a optimizer algotith.

#### 2. Attempts to reduce overfitting in the model

Several probes were mede in order to avoid overfitting. 
After training the model, the testing phase with the simulator was continued in different situations. The car was controlled to run both in a clockwise direction and in situations close to the highway lines. The final model revel

After training the model, the testing phase with the simulator was continued in different situations. The car was controlled to run both in a clockwise direction and in situations close to the highway lines. The final model revealed a robust model thanks to the number of samples used in the training.

Keras callbacks was a tool used to avoid overfitting. With the functions of EarlyStopping and ModelCheckpoint two things are achieved: That the model stops when realizing that there are no better results in the optimizer and that the best result found until the end of the training is saved.

Through several tests it was concluded that with 1 epoch in the training process the model achieved good results, so finally this was the final parameter. So for this model, with such a reduced epochs, keras callback tools are not very useful. However with trained models, witch several epochs, it was seen the potential of these tools to reduce overfitting.
Finally and implicitly the best way to avoid overfitting was providing a diversed and large dataset to the model to be trained. This guaranteed a robust model.

#### 3. Model parameter tuning

The model used an ADAM optimizer, so the learning rate was not tuned manually.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. It was used a combination of center lane driving, center flipped lane driving,  left and right sides of the road. So, the final dataset was 4 times bigger than the firt model trained. 
The differences in the data increase were decisive to have robust results.


### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The dataset is basically an input (image from the front of the car) with its respective steering (angle). The data set was split to have 80% of the imagen for the training and the 20% for validation of the model.  

As different configurations were tested, increasing data was chosen as the best solution to achieve a robust model.
The model used was explained before. It was the Nvidia Model created for self driving cars.

The Nvidia team explains that the Convolutional layers extrat the features of the images (taken from the front of the car running) and the fully connected layers have the function to control the steering of the car. One huge important thing is that they accept that at the end they can not distiguish which layers do exactly what. This point revails the black box character the CNN could have.

Nvidia recommends use the mean squared error(MSE) as loss function and ADAM as optimizer.
First of all the dataset have to be normalized in order to make  normathe trainig faster and easier, the the convolutional layers continued the model.
Each Convoluational layer has its own kernel size but all the conv layers have the same activation function : 'RELU'
At the end there are the fully connected layers
Tha model was trained with one epoch using the Keras callbacks, also.

Finally the model was probed on the road with the simulator. And for the report there race was recorded and saved in a video.
This race shows a robust model. The model was probed with another track and it did not works thar welll, because the images used to train the model a the roads are quite different.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

It is important to name the batch taining (generator) used to avoid load in RAM all the data set. This function do not replace functions like Batch Normalization but helps to train model with a large dataset. 

#### 2. Final Model Architecture

The final model architecture is the same as the first time proposal at the beginning of this report.
This is the summary of the trained model that exposes its architecture. This summary exposes each layer and its respective input.

![FinalModel][image2]



#### 3. Creation of the Training Set & Training Process

In the last project, the personal goal was to modify the dataset as little as possible until the model's potential has been fully exploited. For this project, the opposite was evaluated. The possibilities were to modify the dataset, applying pre-processing techniques. The second possibility was to make the most of the data, using data augmentation. The two possibilities were evaluated, however the best results were obtained by means of data augmetation.

It was decided to use the dataset provided by udacity. Different models were trained using different data set pate.
In principle, a dataset composed of:
* Image taken from the central view
* Flipped image taken from the central view 
* Image taken from the left view
* Image taken from the right view

After the collection process, the dataset has 25712 samples. Then only pre-process aplied was a channel transformation.
Other alternatives such as color transformations, contrast enhancement, histogram equalizations were tested. However, there was no big improvement in the model.

The data set was split 80% for training and 20% validation. Siempe was taken into account shuffle the data. 

The validation was crucial to probe the best performance of the model. Just a few number of epochs were probed because of the total time that takes train this model with that dataset. Finally 1 epoch was enough to have a good model.

[Final Video](run2.mp4)
