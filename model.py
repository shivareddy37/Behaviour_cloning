#imports
import readingCsv
import img_manipulator
import udacity_data_reading
import json
import numpy as np
import cv2

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from keras.callbacks import ModelCheckpoint
from keras.layers import Convolution2D,Cropping2D,MaxPooling2D
from keras.layers.core import Lambda, Dropout
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers.core import Dense, Flatten




OPTIMIZER = Adam(lr=.0001) #.0001 1e-5
LOSS = 'mse'
BATCH_SIZE = 1000
EPOCHS = 10


# model NVDIA 5 conv layers and 4 fully connceted layers
# I included additional dropout layers
model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((70,25), (0,0))))
model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(50, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(10, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(1, activation='tanh'))
model.compile(loss=LOSS, optimizer=OPTIMIZER,metrics=['accuracy'])

print('loading imgages for model traning')
path_left_cam_imgs, path_right_cam_imgs, path_center_cam_imgs, steering_angles = readingCsv.main()
path_to_imgs = path_left_cam_imgs + path_center_cam_imgs + path_left_cam_imgs
steering_angles += (steering_angles + steering_angles)




X_train_path, X_validation_path, y_train_path, y_validation_path = train_test_split(path_to_imgs, steering_angles, test_size=0.20,                                                                      random_state=42)


# python genrator
def generate_data(data,labels, batch_size):
    while 1:
        for offset in range(0, len(data), batch_size):
            batch_data = data[offset:offset + batch_size]
            batch_labels = labels[offset:offset+batch_size]
            images = []
            angles = []

            for path, angle in zip(batch_data, batch_labels):
                src = cv2.imread(path)
                images.append(src)
                angles.append(float(angle))

                flipped = img_manipulator.flip_img(src)
                images.append(flipped)
                angles.append(float(angle) * -1)


                rand_bright = img_manipulator.bright_image(src)
                images.append(rand_bright)
                angles.append(float(angle))

            X = np.array(images)
            Y = np.array(angles)

            yield shuffle(X, Y)


training_data = generate_data(X_train_path,y_train_path, BATCH_SIZE)
validation_data = generate_data(X_validation_path, y_validation_path, BATCH_SIZE)

#
print('Model is training this may take some time')
history_object = model.fit_generator(training_data,
                              samples_per_epoch=len(X_train_path),
                              nb_epoch=EPOCHS,
                              verbose=1,
                              validation_data=validation_data,
                              nb_val_samples=len(X_validation_path))

print('traning completed')

# saving completed
model.save('model.h5')
json_string = model.to_json()
json.dump(json_string, open('model.json', 'w'))
print('model saved')

plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()


print('Trained model file is saved to disc')

