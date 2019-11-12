
import keras
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.preprocessing import image
from keras.engine import Layer
from keras.applications.inception_resnet_v2 import preprocess_input
from keras.layers import Conv2D, UpSampling2D, InputLayer, Conv2DTranspose, Input, Reshape, merge, concatenate, Activation, Dense, Dropout, Flatten
from keras.layers.normalization import BatchNormalization
from keras.callbacks import TensorBoard 
from keras.models import Sequential, Model
from keras.layers.core import RepeatVector, Permute
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from skimage.color import rgb2lab, lab2rgb, rgb2gray, gray2rgb, rgb2hsv, hsv2rgb
from keras.callbacks import ModelCheckpoint
from skimage.transform import resize
from skimage.io import imsave
import numpy as np
import os
import random
import glob
from tqdm import tqdm
from matplotlib import pyplot as plt
import tensorflow as tf
from PIL import Image
from scipy import misc
import sys
import colorsys
from math import ceil
# from transform import convertToLMS, ConvertToDeuteranopes, convertToRGB, normalise, ConvertToTritanopes, ConvertToProtanopes
from daltonize import transform_colorspace, simulate, simulate_array
import h5py

batch_size = 10
epochs = 1

colorblindness = 2

conversion_types = ['gray', 'protanopia', 'deuteranopia', 'tritanopia']
conversion = conversion_types[colorblindness]
daltonize_types = [None, 'p', 'd', 't']
daltonize = daltonize_types[colorblindness]

# def colorblind_array(img):
#     img = img.rotate(-90)
#     size = np.shape(img)[0]
#     img = convertToLMS(img, size, size)
#     if conversion == 'protanopia':
#         img = ConvertToProtanopes(img, size, size)
#     if conversion == 'deuteranopia':
#         img = ConvertToDeuteranopes(img, size, size)
#     if conversion == 'tritanopia':
#         img = ConvertToTritanopes(img, size, size)
#     img = convertToRGB(img, size, size)
#     img = rgb2lab(img)
#     return img


def get_X():
    r = []
    im_dir = 'Train_small'
    r = glob.glob(os.path.join(im_dir, '*.jpg'))
    img_array = []
    for entry in tqdm(r, total=len(r)):
        try:
            im = misc.imread(entry)
            img_array.append(im)
        except:
            print("this entry did not work:" + entry)
    X = np.array(img_array, dtype=float)
    X = 1.0/255*X
    return X

# def get_Ytrain(conversion):
#     r = []
#     #im_dir = 'SUN2012_no_subfolders'
#     if conversion == 'protanopia':
#         im_dir = 'SUN2012_deuteranopia'
#     if conversion == 'deuteranopia':
#         im_dir = 'SUN2012_protanopia'
#     if conversion == 'tritanopia':
#         im_dir = 'SUN2012_tritanopia'
#     else:
#         return
#     r = glob.glob(os.path.join(im_dir, '*.png'))
#     img_array = []
#     for entry in tqdm(r, total=len(r)):
#         try:
#             im = misc.imread(entry)
#             img_array.append(im)
#         except:
#             print("this entry did not work:" + entry)
#     Y = np.array(img_array, dtype=float)
#     Y = 1.0/255*Y
#     return Y

#Ytrain = get_Ytrain(conversion = conversion)



# # Get images
# X = []
# for filename in os.listdir('../Train/'):
#     X.append(img_to_array(load_img('../Train/'+filename)))
# X = np.array(X, dtype=float)

# Set up training and test data
# X = get_X()
# split = int(0.95*len(X))
# Xtrain = X[:split]
# Xtrain = 1.0/255*Xtrain

#Design the neural network
model = Sequential()
if conversion == 'gray':
    model.add(InputLayer(input_shape=(256, 256, 1)))
else:
    model.add(InputLayer(input_shape=(256, 256, 2)))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same', strides=2))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same', strides=2))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same', strides=2))
model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(2, (3, 3), activation='tanh', padding='same'))
model.add(UpSampling2D((2, 2)))

# Finish model
model.compile(optimizer='rmsprop', loss='mse', metrics = ['accuracy'])

# Image transformer
datagen = ImageDataGenerator(
        shear_range=0.2,
        zoom_range=0.2,
        rotation_range=20,
        horizontal_flip=True)

# Generate training data
# def image_a_b_gen(batch_size):
#     for batch in datagen.flow(Xtrain, batch_size=batch_size):
#         lab_batch = rgb2lab(batch)
#         X_batch = lab_batch[:,:,:,0]
#         Y_batch = lab_batch[:,:,:,1:] / 128
#         yield (X_batch.reshape(X_batch.shape+(1,)), Y_batch)

# def image_a_b_gen_train(batch_size):
#     for batch in datagen.flow(Xtrain, batch_size=batch_size):
#         split_batch = int(0.90*len(batch))
#         lab_batch = rgb2lab(batch)
#         if conversion == 'gray':
#             X_batch = lab_batch[:,:,:,0]
#             Y_batch = lab_batch[:,:,:,1:] / 128
#         else:
#             colorblind_batch = batch[:,:,:,0] * 255
#             Y_batch = batch[:,:,:,0] * 255
#             colorblind_whole = simulate_array(batch, daltonize)
#             for i in range(np.shape(colorblind_batch)[0]):
#                 colorblind_piece = colorblind_whole[i]
#                 hsv_piece_colorblind = rgb2hsv(colorblind_piece)
#                 # hsv_piece_colorblind = lab2hsv(colorblind_piece)
#                 X_piece = hsv_piece_colorblind[:,:,0]
#                 Y_peice = rgb2hsv(batch[i])
#                 Y_batch[i] = Y_peice[:,:,0]
#                 colorblind_batch[i] = X_piece
#             X_batch = colorblind_batch
#             # X_batch = lab_batch_colorblind[:,:,:,1:]
#             # Y_batch = lab_batch[:,:,:,1:] / 128
#         X_batch_train = X_batch
#         # [:split_batch]
#         Y_batch_train = Y_batch
#         # [:split_batch]
#         X_batch_train = X_batch_train.reshape(X_batch_train.shape + (1,))
#         Y_batch_train = Y_batch_train.reshape(Y_batch_train.shape + (1,))
#         yield (X_batch_train, Y_batch_train) ###+(1,))
# def image_a_b_gen_train(batch_size):
#     for batch in datagen.flow(Xtrain, batch_size=batch_size):
#         split_batch = int(0.90*len(batch))
#         lab_batch = rgb2lab(batch)
#         if conversion == 'gray':
#             X_batch = lab_batch[:,:,:,0]
#             Y_batch = lab_batch[:,:,:,1:] / 128
#         else:
#             colorblind_batch = X_batch * 255
#             colorblind_batch = simulate_array(colorblind_batch, daltonize)
#             lab_batch_coloblind = rgb2lab(colorblind_batch)
#             X_batch = lab_batch_colorblind[:,:,:,1:]
#             Y_batch = lab_batch[:,:,:,1:]
#         X_batch_val = X_batch[split_batch:]
#         X_batch_val = X_batch[split_batch:]
#         yield (X_batch_val.reshape(X_batch_val.shape), Y_batch_val) ###+(1,))

#### MY CODE 
# def image_a_b_gen_val(batch_size):
#     for batch in datagen.flow(Xtrain, batch_size=batch_size):
#         split_batch = int(0.90*len(batch))
#         lab_batch = rgb2lab(batch)
#         if conversion == 'gray':
#             X_batch = lab_batch[:,:,:,0]
#             Y_batch = lab_batch[:,:,:,1:] / 128
#             X_batch_val = X_batch[:split_batch]
#             X_batch_val = np.reshape(X_batch_val, X_batch_val.shape + (1,))
#         else:
#             colorblind_batch = batch
#             colorblind_batch = simulate_array(colorblind_batch, daltonize)
#             lab_batch_colorblind = rgb2lab( colorblind_batch)
#             X_batch = lab_batch_colorblind[:,:,:,1:]
#             Y_batch = lab_batch[:,:,:,1:]
#             X_batch_val = X_batch[:split_batch]
#         Y_batch_val = Y_batch[:split_batch]
#         yield (X_batch_val.reshape(X_batch_val.shape), Y_batch_val) ###+(1,))


 
#### CRISTIAN"S CODE



train_path = "sun6_dataset.hdf5"
train_file = h5py.File(train_path,"r")
x_train = train_file["test_img"]
y_train = train_file["test_img"]

X = x_train

def generate_batch(batch_size = 10, batch_ind = 0):
    # loop over batches   
    i_s = batch_ind * batch_size
    i_e = min((batch_ind +1)*batch_size, number)
    return train_file["test_img"][i_s:i_e, ...]


def image_a_b_gen(batch_size):
    num_batches = ceil(number/batch_size)
    for i in range(0, int(num_batches)):
        batch = generate_batch(batch_size = batch_size, batch_ind = i)
        split_batch = int(0.90*len(batch))
        batch = batch *  1/255
        lab_batch = rgb2lab(batch)
        if conversion == 'gray':
            X_batch = lab_batch[:,:,:,0]
            Y_batch = lab_batch[:,:,:,1:] / 128
        else:
            colorblind_batch = batch
            colorblind_batch = simulate_array(colorblind_batch, daltonize)
            lab_batch_colorblind = rgb2lab( colorblind_batch)
            X_batch = lab_batch_colorblind[:,:,:,1:]
            Y_batch = lab_batch[:,:,:,1:]
        X_batch_val = X_batch[:split_batch]
        Y_batch_val = Y_batch[:split_batch]
        yield (X_batch_val.reshape(X_batch_val.shape), Y_batch_val) 

number, height, width, channels = np.shape(x_train)
steps_per_epoch = number / batch_size
validation_steps = 50
####
# , validation_steps = validation_steps, validation_data = image_a_b_gen_val(batch_size)

# Train model
TensorBoard(log_dir='/output')
history = model.fit_generator(image_a_b_gen(batch_size), steps_per_epoch= steps_per_epoch, epochs= epochs)


# # Test images
# if conversion == 'gray':
#     Xtest = rgb2lab(1.0/255*X[split:])[:,:,:,0]
#     Xtest = Xtest.reshape(Xtest.shape+(1,))
#     Ytest = rgb2lab(1.0/255*X[split:])[:,:,:,1:]
#     Ytest = Ytest / 128
# else:
#     # Xtest = rgb2lab(simulate_array( 1.0/255*X[split:] , daltonize))[:,:,:,1:]
#     Xtest = rgb2lab(simulate_array( 1.0/255*X[split:] , daltonize))[:,:,:,1:]
#     Xtest = Xtest.reshape(Xtest.shape) 
#     # +(1,)
#     Ytest = rgb2lab(1.0/255*X[split:])[:,:,:,1:]
#     # Ytest = rgb2lab(1.0/255*X[split:])[:,:,:,1:]
#     # Ytest = Ytest / 128
# print model.evaluate( Xtest, Ytest, batch_size=batch_size)


plt.plot(history.history['acc'])
#plt.plot(history.history['val_acc'])
plt.title('model accuracy for ' + str(conversion))
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
# , 'validation'
plt.savefig('/Users/alexlawson/Documents/CLPS1950_python/Color_Recognition_Project/Colorizer_Main/result_' + str(conversion) + '/model_' + str(conversion) + '.png')
#our_model.save('/home/mjthomps/course/CLPS/datathon_model6.h5')

def get_Xtest():
    r = []
    im_dir = 'Test_last_20'
    r = glob.glob(os.path.join(im_dir, '*.jpg'))
    img_array = []
    for entry in tqdm(r, total=len(r)):
        try:
            im = misc.imread(entry)
            # im = gray2rgb(rgb2gray(im))
            img_array.append(im)
        except:
            print("this entry did not work:" + entry)
    X = np.array(img_array, dtype=float)
    x = 1.0/255 * X
    return X

# Load black and white images
# color_me = []
# for filename in os.listdir('../Test_last_20'):
#         color_me.append(img_to_array(load_img('../Test_last_20/'+filename)))
if conversion == 'gray':
    color_me = get_Xtest()
    color_me = rgb2lab(1.0/255*color_me)[:,:,:,0]
    color_me = color_me.reshape(color_me.shape+(1,))
    output = model.predict(color_me)
    output = output * 128
else:
    color_me = get_Xtest()
    color_me_blind = rgb2lab(simulate_array(color_me, daltonize))[:,:,:,1:]
    color_me_blind = color_me_blind.reshape(color_me_blind.shape)
    print(np.shape(color_me_blind))
    # color_me_blind = rgb2lab(simulate_array( 1.0/255*X[split:] , daltonize))[:,:,:,1:]
    # color_me_blind = color_me_blind.reshape(color_me_blind.shape)
    # +(1,)
    output = model.predict(color_me_blind)
    output = output * 255
    print(np.shape(output))

# Test model


# Output colorizations

for i in range(np.shape(color_me)[0]):
        cur = np.zeros((256, 256, 3))
        cur[:,:,0] = color_me[i][:,:,0]
        cur[:,:,1:] = output[i]
        imsave("result_" + str(conversion) + "/img_" +str(format(i, '02d'))+" .jpg", lab2rgb(cur))



