import cv2
import numpy as np
import csv
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from keras.models import Sequential
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Flatten, Dense, Lambda, MaxPooling2D, Dropout, Convolution2D, Cropping2D
 
import numpy as np
import csv
import cv2
import sklearn

from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, MaxPooling2D, Dropout
from keras.layers.convolutional import Convolution2D

## Pre-process function
def pre_process_image(image):
    #Change format from BGR to RGB
    colored_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #Modify the scale to make the model faster in learning module
    #resized_image = cv2.resize(cropped_image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
    return colored_image

## Network Model
def nvidiaModel():
    # Model based on Nvidia network      
    model = Sequential()
    # Normalization
    model.add(Lambda(lambda x: (x / 255.0) - 0.5 , input_shape=processed_image_shape))    
    # Crop the image, to used just the important part.
    model.add(Cropping2D(cropping=((50,20), (0,0))))
    #Network
    model.add(Convolution2D(24,5,5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(36,5,5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(48,5,5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(64,3,3, activation='relu'))
    model.add(Convolution2D(64,3,3, activation='relu'))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    
    # Alternative Model
    # model = Sequential()
    # model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=processed_image_shape))
    # #In case to set the shape of the image manually
    # #model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(32,80,3)))
    # model.add(Conv2D(24, (5,5), activation='relu'))
    # #model.add(BatchNormalization())
    # model.add(Conv2D(36, (5, 5), activation='relu'))
    # #model.add(BatchNormalization())
    # model.add(Conv2D(48, (5, 5), activation='relu'))
    # #model.add(BatchNormalization())
    # model.add(Conv2D(64, (3, 3), activation='relu'))
    # #model.add(BatchNormalization())
    # model.add(Conv2D(64, (3, 3), activation='relu'))
    # #model.add(BatchNormalization())
    # model.add(Flatten())
    # model.add(Dense(100))
    # #model.add(BatchNormalization())
    # #model.add(Dropout(0.25))
    # model.add(Dense(50))
    # #model.add(BatchNormalization())
    # #model.add(Dropout(0.25))
    # model.add(Dense(10))
    # #model.add(BatchNormalization())
    # #model.add(Dropout(0.25))
    # model.add(Dense(1))
    return model

## Gnerator created to define the trainig through batches
def generator(input_data, image_path, batch_size=32, left_image_angle_correction = 0.20, right_image_angle_correction = -0.20):
    #Create batch to process
    processing_batch_size = int(batch_size)
    number_of_entries = len(input_data)
    
    while 1: 
        #Always
        for offset in range(0, number_of_entries, processing_batch_size):
            #Processing each sample of the batch
            #Define the batch data
            batch_data = input_data[offset:offset + processing_batch_size]      
            images = []
            angles = []
            for batch_sample in batch_data:
                ##Process to each sample of the batch
                #First take the center image and its angle
                path_center_image = image_path+(batch_sample[0].strip()).split('/')[-1]
                angle_for_centre_image = float(batch_sample[3])
                center_image = cv2.imread(path_center_image)
				
                #Check if the image is OK
                if center_image is not None:
                    # Pre-process the center image
                    processed_center_image = pre_process_image(center_image)
                    #And start to populate the vector of images
                    images.append(processed_center_image)
                    angles.append(angle_for_centre_image)
                    ## Flipping the image
                    images.append(cv2.flip(processed_center_image, 1))
                    angles.append(-angle_for_centre_image)
				                
                #IMPORTANT- For the Dataset will be used the left, right and center view, as well as the flipping center view
				#In case needed flipp the left and the right view                 
				## Pre-process the left image
                left_image_path = image_path + batch_sample[1].split('/')[-1]
                left_image = cv2.imread(left_image_path)
                if left_image is not None:
                    images.append(pre_process_image(left_image))
                    angles.append(angle_for_centre_image + left_image_angle_correction)

                # Pre-process the right image
                right_image_path = image_path + batch_sample[2].split('/')[-1]
                right_image = cv2.imread(right_image_path)
                if right_image is not None:
                    images.append(pre_process_image(right_image))
                    angles.append(angle_for_centre_image + right_image_angle_correction)

			# Shuffling and returning the image data back to the calling function
            yield sklearn.utils.shuffle(np.array(images), np.array(angles))

# Define Constants and Paths
data_path = "data/"
image_path = data_path + "IMG/"
csv_data = []
processed_csv_data = []
csvPath = 'data/driving_log.csv'

#Loading Data (CSV reference from the data)
with open(csvPath) as csv_file:
    csv_reader = csv.reader(csv_file)
    # Skipping the headers
    next(csv_reader, None)
    for each_line in csv_reader:
        csv_data.append(each_line)

# Shuffle the csv entries and split the train and validation dataset
csv_data = sklearn.utils.shuffle(csv_data)
train_samples, validation_samples = train_test_split(csv_data, test_size=0.2)
#Datasets
train_generator = generator(train_samples, image_path)
validation_generator = generator(validation_samples, image_path)

#Get the shape of the process
first_img_path = image_path + csv_data[0][0].split('/')[-1]
first_image = cv2.imread(first_img_path)
processed_image_shape = pre_process_image(first_image).shape
print (processed_image_shape)

#Compile the model
model = nvidiaModel()
model.compile(optimizer= 'adam', loss='mse', metrics=['acc'])

# Name of the model to save
file = 'model.h5'
##Define some features to save time in the training and save the beset result
#Stop training in case of no improvement
stopper = EarlyStopping(patience=5, verbose=1)
#Save the best model 
checkpointer = ModelCheckpoint(file, monitor='val_loss', verbose=1, save_best_only=True)


print("Trainning")
epoch = 1
history_object = model.fit_generator(train_generator, 
									 samples_per_epoch = 4*len(train_samples),
                                     validation_data = validation_generator,
                                     nb_val_samples = 4*len(validation_samples), 
									 nb_epoch=epoch, 
									 verbose=1)
#saving model
print("Saving model")
model.save(file)
print("Model Saved")
# keras method to print the model summary
model.summary()    
##Take some information about the training and the final model
print(history_object.history.keys())
print('Loss')
print(history_object.history['loss'])
print('Validation Loss')
print(history_object.history['val_loss'])

#Plot results validation_loss and train_loss
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
#plt.show()
plt.savefig('Train_Valid_Loss.png')
