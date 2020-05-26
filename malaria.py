from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.models import Sequential
from keras.models import load_model
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

m = load_model('Malarial-Cell-Prediction-Model.h5')

model = Sequential()

model.add(Convolution2D(filters=32,kernel_size=(3,3),  activation='relu',input_shape=(28, 28, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))


model.add(Flatten())

model.add(Dense(units=128, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)
training_set = train_datagen.flow_from_directory(
        'cell_images/training_set/',
        target_size=(28, 28),
        batch_size=32,
        class_mode='binary')
test_set = test_datagen.flow_from_directory(
        'cell_images/testing_set/',
        target_size=(28, 28),
        batch_size=32,
        class_mode='binary')
model.fit(
        training_set,
        steps_per_epoch=3000,
        epochs=5,
        validation_data=test_set,
        validation_steps=800)


test_image = image.load_img('cell_images/prediction_set/I3.png',target_size=(28,28))
test_image = image.img_to_array(test_image)
test_image.shape
test_image = np.expand_dims(test_image, axis=0)
test_image.shape

result = m.predict(test_image)

if result[0][0] == 1.0:
    print('Uninfected')
else:
    print('Parasitized')

r = training_set.class_indices #tells no. of infected or unifected images predicted

#**************************

acc=model.evaluate(training_set)
accuracy=str(acc[1])
f = open("k.txt", "w")
f.write(accuracy)
f.close()


model.save('Malarial-Cell-Prediction-Model.h5')

