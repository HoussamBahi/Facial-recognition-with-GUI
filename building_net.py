import cv2
import numpy as np
import requests
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sys
import datetime

from keras.engine.saving import load_model
from keras.preprocessing import image
from keras_preprocessing.image import img_to_array
from tensorflow import keras
from tensorflow.keras.models import Model
import tensorflow as tf



#Import dataset

import pathlib
import os

path= 'model2/'
data_dir = 'images/'



data_dir = pathlib.Path('images/')
print(data_dir)
print(os.path.abspath(data_dir))

image_count = len(list(data_dir.glob('*/*')))
print(image_count)


batch_size = 64
img_height = 200
img_width = 200



# training dataset
train_data = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=42,
  image_size=(img_height, img_width),
  batch_size=batch_size,
  )

# validation dataset

val_data = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=42,
  image_size=(img_height, img_width),
  batch_size=batch_size
)
# test dataset

test_data= tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=42,
  image_size=(img_height, img_width),
  batch_size=batch_size)



# print the labels ( name of images )

class_names = val_data.class_names
print(class_names)



# show 3 images samples random
#plt.figure(figsize=(10, 10))
# for images, labels in train_data.take(1):
#   for i in range(3):
#     ax = plt.subplot(1, 3, i + 1)
#     plt.imshow(images[i].numpy().astype("uint8"))
#     plt.title(class_names[labels[i]])
#     plt.axis("off")


from tensorflow.keras import layers

# lables or classes
num_classes = 5
# model structure base CNN
model = tf.keras.Sequential([
    layers.experimental.preprocessing.Rescaling(1./255),
    layers.Conv2D(128,4, activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.5),
    layers.Conv2D(64,4, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32,4, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(16,4, activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(64,activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])
# Compiling the model
model.compile(optimizer='adam',
              loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['accuracy'],)

logdir="logs"

#tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir,histogram_freq=1, write_images=logdir,
#                                                  embeddings_data=train_data)
# Training the model
acc_train1 = []
acc_val1 = []
history= model.fit(
    train_data,
    #validation_split=0.2,
    validation_data=val_data,
    epochs=100
)
acc_train1 += history.history['accuracy']
acc_val1 += history.history['val_accuracy']

# Saving the model
#todo VERY IMPORTANT after each train we have to change the name of the saved model so the old won't get overwritten
# model.save_weights(path + '_final_weights.h5')
# model.save(path + '_my_model.h5')  # creates a HDF5 file 'my_model.h5'
# model_json = model.to_json()

print("Saving final model...")

model.save_weights(path + '_final_weights.h5')
model.save(path + '_my_model1.h5')  # creates a HDF5 file 'my_model.h5'
model_json = model.to_json()

with open(path + "_model.json", "w") as json_file:
    json_file.write(model_json)

#model.save(path + 'mmodel.h5')

model.summary()





#mmodel= load_model("C:/Users//houssem//PycharmProjects//FR//model2//_my_model.h5")

## image externe pour tester notre model

# test image from our dataset of our choice
# in this case jack12
# testimm= image.load_img('images/jack/jack12.jpg')
# # testimm= img_to_array(testimm)
# # testimm=np.expand_dims(testimm, axis=0)
# # res= model.predict(testimm)
# # ress=model.predict_classes(testimm)
# # resss= np.argmax(model.predict(testimm), axis=-1)
# # print(ress)


plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 18

plt.plot(acc_train1, linewidth=2)
plt.plot(acc_val1, linewidth=2)

plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.grid()

plt.show()
#testing our model on a testdata to get the accuracy and loss
print("Plotting accuracy versus epoch")
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.show()
print("The model is being evaluated")

#evaluate our model to calculate the accuracy and the loss evaluer notre modele pour calculer le taux et la perte

test_loss, test_acc = model.evaluate(test_data, verbose=2)
print("The accuracy of the model is:")

# print accuracy and loss
print(test_acc)
print(test_loss)








