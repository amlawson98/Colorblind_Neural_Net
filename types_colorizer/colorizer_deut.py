

from keras.layers import Conv2D, UpSampling2D, InputLayer
from keras.callbacks import TensorBoard 
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from skimage.color import rgb2lab, lab2rgb, rgb2gray, gray2rgb, rgb2hsv, hsv2rgb
from skimage.io import imsave
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
import tensorflow as tf
# from PIL import Image
import glob as glob
import os
from math import ceil
# from transform import convertToLMS, ConvertToDeuteranopes, convertToRGB, normalise, ConvertToTritanopes, ConvertToProtanopes
from daltonize import transform_colorspace, simulate, simulate_array
import h5py


# def parse_args():
#     parser = argparse.ArgumentParser(prog = 'colorizer.py')
#     parser.add_argument('-clr', type = int)
#     parser.add_argument('-epochs', type = int)
#     parse.add_argument('-batch_size', type = int)
#     args = parser.parse_args()
#     return args

# if args.clr == None:
colorblindness = 2
# else:
#     colorblindness = args.clr

# if args.epochs == None:
epochs = 5
# else:
#     epochs = args.epochs

# if args.batch_size == None:
batch_size = 20
# else:
#     batch_size = args.batch_size


conversion_types = ['gray', 'protanopia', 'deuteranopia', 'tritanopia']
conversion = conversion_types[colorblindness]
daltonize_types = [None, 'p', 'd', 't']
daltonize = daltonize_types[colorblindness]


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




train_path = "sun2012_dataset.hdf5"
train_file = h5py.File(train_path,"r")
x_train = train_file["train_img"]
x_val = train_file['val_img']

X = x_train

def generate_batch_train(batch_size = batch_size, batch_ind = 0):
    # loop over batches   
    i_s = batch_ind * batch_size
    i_e = min((batch_ind +1)*batch_size, number)
    return train_file["train_img"][i_s:i_e, ...]

def image_a_b_gen_train(batch_size):
    num_batches = ceil(number/batch_size)
    for i in range(0, int(num_batches)):
        batch = generate_batch_train(batch_size = batch_size, batch_ind = i)
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
        yield (X_batch.reshape(X_batch.shape), Y_batch) 



def generate_batch_val(batch_size = batch_size, batch_ind = 0):
    # loop over batches   
    i_s = batch_ind * batch_size
    i_e = min((batch_ind +1)*batch_size, number)
    return train_file["val_img"][i_s:i_e, ...]

def image_a_b_gen_val(batch_size):
    num_batches = ceil(number/batch_size)
    for i in range(0, int(num_batches)):
        batch = generate_batch_val(batch_size = batch_size, batch_ind = i)
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
        yield (X_batch.reshape(X_batch.shape), Y_batch) 


number, height, width, channels = np.shape(x_train)
steps_per_epoch = number / batch_size


validation_steps = len(x_val[0])/ batch_size
# Train model
TensorBoard(log_dir='/output')

history = model.fit_generator(
    image_a_b_gen_train(batch_size),
    steps_per_epoch= steps_per_epoch,
    epochs= epochs,
    validation_data = image_a_b_gen_val(batch_size),
    validation_steps = validation_steps)






plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
#plt.plot(history.history['val_acc'])
plt.title('model accuracy for ' + str(conversion))
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
# , 'validation'
plt.savefig('result_' + str(conversion) + '/model_' + str(conversion) + '.jpg')
#our_model.save('/home/mjthomps/course/CLPS/datathon_model6.h5')

def get_Xtest():
    r = []
    im_dir = 'sun6_copy2'
    r = glob.glob(os.path.join(im_dir, '*.png'))
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

# if __main__ == '__main__':
#     args = parse_args()
