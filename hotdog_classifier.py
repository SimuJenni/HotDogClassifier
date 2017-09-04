from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
import matplotlib.pyplot as plt

"""
    Prepare the data for training and testing
"""

train_dir = '../seefood/train'
test_dir = '../seefood/test'

batch_size = 16
target_size = (64, 64)
classes = ['hot_dog', 'pizza']

# This data-generator performs different forms of augmentation
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.,
    zoom_range=0.,
    rotation_range=0.,
    horizontal_flip=False)

# Only rescale the images for testing
test_datagen = ImageDataGenerator(rescale=1. / 255)

# This generator will read images from the specified folder and create minibatches
train_generator = train_datagen.flow_from_directory(
    train_dir,  # this is the target directory
    target_size=target_size,  # images will be resized
    batch_size=batch_size,
    classes=classes,  # the classes used for training
    class_mode='binary')  # since we use binary_crossentropy loss, we need binary labels

validation_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=target_size,
    batch_size=batch_size,
    classes=classes,
    class_mode='binary')

"""
    Build the neural network
"""

model = Sequential()

# Convolution layers (typically conv2d -> activation -> pooling
model.add(Conv2D(filters=32, kernel_size=(3, 3), input_shape=target_size + (3,)))  # input_shape only for 1st layer
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Fully-connected layers
model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(units=64))
model.add(Activation('relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(units=1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

"""
    Training the network
"""

history = model.fit_generator(
    train_generator,
    steps_per_epoch=2000 // batch_size,
    epochs=30,
    validation_data=validation_generator,
    validation_steps=800 // batch_size,
    workers=4,
    use_multiprocessing=True)
model.save_weights('first_try.h5')  # always save your weights after training or during training

"""
    Visualize training
"""

print(history.history.keys())

plt.figure(1, figsize=(10, 10))

# summarize history for accuracy
plt.subplot(211)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')

# summarize history for loss

plt.subplot(212)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


"""
    Show some sample predictions
"""

test_x, test_y = test_datagen.flow_from_directory(test_dir,
                                                  target_size=target_size,
                                                  batch_size=16,
                                                  classes=classes,
                                                  class_mode='binary').next()
pred = model.predict(test_x, batch_size=16)
nx, ny = (2, 8)
f, axarr = plt.subplots(2, 8, figsize=(20, 5))
for x in range(nx):
    for y in range(ny):
        idx = x * ny + y
        axarr[x, y].imshow(test_x[idx])
        axarr[x, y].set_title('Pred: {}'.format(pred[idx]))
plt.show()
