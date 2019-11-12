"""Loads all images. Converts all images to all kinds of colorblindness 
and to grayscale in HSV coordinates."""


import h5py
from skimage.color import rgb2hsv, hsv2rgb, rgb2lab, lab2rgb
from matplotlib import pyplot as plt
import numpy as np
import os




def transform_colorspace(img, mat):
    """Transform image to a different color space.
    Arguments:
    ----------
    img : array of shape (M, N, 3)
    mat : array of shape (3, 3)
        conversion matrix to different color space
    Returns:
    --------
    out : array of shape (M, N, 3)
    """
    # Fast element (=pixel) wise matrix multiplication
    return np.einsum("ij, ...j", mat, img)


def simulate(img, color_deficit="d"):
    """Simulate the effect of color blindness on an image.
    Arguments:
    ----------
    img : PIL.PngImagePlugin.PngImageFile, input image
    color_deficit : {"d", "p", "t"}, optional
        type of colorblindness, d for deuteronopia (default),
        p for protonapia,
        t for tritanopia
    Returns:
    --------
    sim_rgb : array of shape (M, N, 3)
        simulated image in RGB format
    """
    # Colorspace transformation matrices
    cb_matrices = {
        "d": np.array([[1, 0, 0], [0.494207, 0, 1.24827], [0, 0, 1]]),
        "p": np.array([[0, 2.02344, -2.52581], [0, 1, 0], [0, 0, 1]]),
        "t": np.array([[1, 0, 0], [0, 1, 0], [-0.395913, 0.801109, 0]]),
    }
    rgb2lms = np.array([[17.8824, 43.5161, 4.11935],
                        [3.45565, 27.1554, 3.86714],
                        [0.0299566, 0.184309, 1.46709]])
    # Precomputed inverse
    lms2rgb = np.array([[8.09444479e-02, -1.30504409e-01, 1.16721066e-01],
                        [-1.02485335e-02, 5.40193266e-02, -1.13614708e-01],
                        [-3.65296938e-04, -4.12161469e-03, 6.93511405e-01]])

    img = img.copy()
    img = img.convert('RGB')

    rgb = np.asarray(img, dtype=float)
    # first go from RBG to LMS space
    lms = transform_colorspace(rgb, rgb2lms)
    # Calculate image as seen by the color blind
    sim_lms = transform_colorspace(lms, cb_matrices[color_deficit])
    # Transform back to RBG
    sim_rgb = transform_colorspace(sim_lms, lms2rgb)
    return sim_rgb


def simulate_array(array, color_deficit):
    """Simulate the effect of color blindness on an image.
    Arguments:
    ----------
    img : PIL.PngImagePlugin.PngImageFile, input image
    color_deficit : {"d", "p", "t"}, optional
        type of colorblindness, d for deuteronopia (default),
        p for protonapia,
        t for tritanopia
    Returns:
    --------
    sim_rgb : array of shape (M, N, 3)
        simulated image in RGB format
    """
    # Colorspace transformation matrices
    cb_matrices = {
        "d": np.array([[1, 0, 0], [0.494207, 0, 1.24827], [0, 0, 1]]),
        "p": np.array([[0, 2.02344, -2.52581], [0, 1, 0], [0, 0, 1]]),
        "t": np.array([[1, 0, 0], [0, 1, 0], [-0.395913, 0.801109, 0]]),
    }
    rgb2lms = np.array([[17.8824, 43.5161, 4.11935],
                        [3.45565, 27.1554, 3.86714],
                        [0.0299566, 0.184309, 1.46709]])
    # Precomputed inverse
    lms2rgb = np.array([[8.09444479e-02, -1.30504409e-01, 1.16721066e-01],
                        [-1.02485335e-02, 5.40193266e-02, -1.13614708e-01],
                        [-3.65296938e-04, -4.12161469e-03, 6.93511405e-01]])
    sim_rgb_full = np.zeros(np.shape(array))
    for i in range(np.shape(array)[0]):
        lms = transform_colorspace(array[i], rgb2lms)
        sim_lms = transform_colorspace(lms, cb_matrices[color_deficit])
        sim_rgb = transform_colorspace(sim_lms, lms2rgb)
        sim_rgb_full[i] = sim_rgb
    return sim_rgb_full


colorblindness = 2
conversion_types = ['gray', 'protanopia', 'deuteranopia', 'tritanopia']
conversion = conversion_types[colorblindness]
daltonize_types = [None, 'p', 'd', 't']
daltonize = daltonize_types[colorblindness]

def unpack_datasets():
    sun2012_path = os.path.join("Data/sun6_dataset.hdf5")
    sun6_path = os.path.join("Data/sun6_dataset_test.hdf5")
    
    sun2012_file = h5py.File(sun2012_path, "r")
    sun6_file = h5py.File(sun6_path, "r")
    
    x_train = sun2012_file["train_img"]  
    y_train = sun2012_file["train_img"] # 
    x_val = sun2012_file["val_img"]
    y_val = sun2012_file["val_img"]
    
    x_test = sun6_file["test_img"]
    y_test = sun6_file["test_img"]
    return x_train, y_train, x_val, y_val, x_test, y_test

def plot_validation_performance(
        train_loss = None,
        val_loss = None,
        colors = ['b', 'r'],
        label = '',
        desc = 'Loss',
        ax = None,
        f = None):
    """Plot losses on a training versus validation set
        Parameters
    ----------
    train_loss : list
        List of losses on the train dataset
    val_loss : list
        List of losses on the validation dataset
    colors : list
        List of colors for plotting
    label : list
        Labels for the plot legends

    Returns
    -------
    None

    """
    if f is None and ax is None:
        f, ax = plt.subplots()
    train_label = '%s training %s' % (label, desc)
    if train_loss is not None:
        ax.plot(train_loss, '%s-' % colors[0], label=train_label)
    ax.set_xlabel('Epoch')
    ax.set_ylabel(desc, color=colors[0])
    val_label = '%s validation %s' % (label, desc)
    if val_loss is not None:
        ax.plot(val_loss, colors[1], label=val_label)
    plt.legend()
    f.tight_layout()
    return f, ax

def plot_test_performance(
        model_losses,
        colors = None,
        label = '',
        desc = ''):
    f, ax = plt.subplots()



def rgb_2_hsv_ar(array):
  for i in range(np.shape(array)[0]):
    array[i] = rgb2hsv(array[i])
  return array

def hsv_2_rgb_ar(array):
    for i in range(array.shape[0]):
        array[i] = hsv2rgb(array[i])
    return array


def preprocess_batch(x_batch, y_batch):
    if conversion == 'gray':
        x_batch = rgb2lab(x_batch)[:,:,:,0] 
        x_batch = x_batch.reshape(np.shape(x_batch) + (1,))
        y_batch = rgb2lab(y_batch)[:,:,:,1:]
    else:
        y_batch = rgb_2_hsv_ar(y_batch)
        y_batch = y_batch[:,:,:,0]
        y_batch = y_batch.reshape(np.shape(y_batch) + (1,))
        x_batch = simulate_array(x_batch, 'd') ########
        x_batch = rgb_2_hsv_ar(x_batch)[:,:,:,0] 
        x_batch = x_batch.reshape(np.shape(x_batch) + (1,))
    return x_batch, y_batch