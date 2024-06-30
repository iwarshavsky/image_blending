import numpy as np
from scipy import ndimage

def getGaussianPyramid(im,pyramid_dimensions):
    """
    Return an array of 2D numpy arrays which are the layers of the Gaussian Pyramid of the given
    image.
    :param im: the input image.
    :param pyramid_dimensions: The dictionary returned from calling getPyramidDimensions on the
    image (the number of entries is the number of layers in the pyramid)
    :return: The array described above.
    """
    reduced_images = [im]
    for i in range(len(pyramid_dimensions)-1):
        reduced_images.append(reduce(reduced_images[i], 1))
    return reduced_images

def getLaplacianPyramid(im,pyramid_dimensions):
    """
    Return an array of 2D numpy arrays which are the layers of the Laplacian Pyramid of the given
    image.
    :param im: the input image.
    :param pyramid_dimensions: The dictionary returned from calling getPyramidDimensions on the
    image (the number of entries is the number of layers in the pyramid).
    :return: The array described above.
    """
    gaussian_pyramid = getGaussianPyramid(im,pyramid_dimensions)

    laplacian_pyramid = [gaussian_pyramid[-1]]
    for i in range(len(gaussian_pyramid) - 2,-1,-1):
        laplacian_pyramid.append(gaussian_pyramid[i]-enlarge(gaussian_pyramid[i+1],
                                                             pyramid_dimensions))
    laplacian_pyramid.reverse()
    return laplacian_pyramid

def hybridImage(im_close, im_far, transition_layer, ignore_layers, global_illumination):
    """
    Creates a hybrid image out of im_close and im_far using their Laplacian pyramids. This is done
    by expanding and summing iteratively the layers of im_far's pyramid until transition_layer is
    reached, afterwhich the layers of im_close will be added.

    :param im_close: A 2D numpy array of the image to be perceived when viewed from up close.
    :param im_far:  A 2D numpy array of the image to perceived when viewed from afar.
    :param transition_layer:  The number of the layer when we start taking into account the other
     pyramid
    :param ignore_layers: Array of layer numbers to be ignored. The sum at the iteration of these
    layers will be expanded without adding anything to it.
    :param global_illumination: Add global illumination to the resulting image.
    :return: A 2D numpy array of the resulting hybrid image.
    """
    # Convert image arrays to float.
    im_close = im_close.astype(float)
    im_far = im_far.astype(float)

    pyramid_dimensions = getPyramidDimensions(im_close)

    # Create Laplacian pyramid for im_close, im_far
    laplacian_pyramid_im_close = getLaplacianPyramid(im_close,pyramid_dimensions)
    laplacian_pyramid_im_far = getLaplacianPyramid(im_far,pyramid_dimensions)

    # Add the layers of im_far's pyramid until transition_layer, then moe to im_close's pyramid.
    # Ignore layers in array ignore_layers.

    laplacian_pyramid_im_far.reverse()
    laplacian_pyramid_im_close.reverse()

    l_sum = laplacian_pyramid_im_far[0]

    for i in range(1,len(pyramid_dimensions)):
        if i in ignore_layers:
            l_sum = enlarge(l_sum,pyramid_dimensions)
        elif i < transition_layer:
            l_sum = enlarge(l_sum,pyramid_dimensions) + laplacian_pyramid_im_far[i]
        else:
            l_sum = enlarge(l_sum, pyramid_dimensions) + laplacian_pyramid_im_close[i]

    return (l_sum + global_illumination).clip(0, 255).astype(np.uint8)

def imageBlending(im1, im2, mask):
    """
    Blend im1 and im2 using the mask - im1 will appear in the result in areas where the mask has
    value 0 (black), im2 will appear in areas where is has value 1 (white).

    :param im1: A 2D numpy array of the first image.
    :param im2: A 2D numpy array of the first image.
    :param mask: A 2D numpy array of the mask.
    :return: A 2D numpy array of the blended image.
    """

    im1 = im1.astype(float)
    im2 = im2.astype(float)
    mask = mask.astype(float)

    pyramid_dimensions = getPyramidDimensions(im1)

    # Create Laplacian pyramid for im1, im2
    laplacian_pyramid_im1 = getLaplacianPyramid(im1,pyramid_dimensions)
    laplacian_pyramid_im2 = getLaplacianPyramid(im2,pyramid_dimensions)

    # Create a Gaussian pyramid for the mask
    mask_gaussian_pyramid = getGaussianPyramid(mask,pyramid_dimensions)

    # Create Laplacian pyramid using mask and im1 and im2's Laplacian pyramids.
    mask_laplacian_pyramid = [(1-mask_gaussian_pyramid[i])*laplacian_pyramid_im1[i]
                              + mask_gaussian_pyramid[i]*laplacian_pyramid_im2[i]
                              for i in range(len(mask_gaussian_pyramid))]

    laplacian_pyramid_mask_reversed = list(reversed(mask_laplacian_pyramid))

    # Sum the Laplacian pyramid up
    l_sum = laplacian_pyramid_mask_reversed[0]
    for i in range(1, len(laplacian_pyramid_mask_reversed)):
        l_sum = enlarge(l_sum,pyramid_dimensions) + laplacian_pyramid_mask_reversed[i]

    return l_sum.clip(0, 255).astype(np.uint8)

def getPyramidDimensions(im):
    """
    Returns a dictionary with keys being the dimensions of every layer in the pyramid and the values
    a boolean tuple whose values dictate whether a row/column of pixels should be added when
    expanding the current layer.

    :param im: the image
    :return: the dictionary previously described.
    """
      # coordinates 3+4 = padding for enlargment of x and y
    cur_dimension = im.shape
    pyramid_dimensions = {}
    bool_from_prev = None
    while 0 not in cur_dimension:
        pyramid_dimensions[cur_dimension]=bool_from_prev
        bool_from_prev = (
                        0 if cur_dimension[0] / 2 == cur_dimension[0] // 2 else 1,
                        0 if cur_dimension[1] / 2 == cur_dimension[1] // 2 else 1)
        cur_dimension = (cur_dimension[0] // 2,
                         cur_dimension[1] // 2)

    return pyramid_dimensions

def reduce(image, times=1, sigma=3):
    """
    Reduce the given image the number times specified after applying a gaussian filter by
    subsampling a pixel on every second row and column, effectively cutting the height and width by
    half.

    Create next gaussian pyramid layer
    :param image:
    :return:
    """
    if (times == 0):
        return image
    im = ndimage.gaussian_filter(image, sigma)[1::2, 1::2]
    if (times == 1):
        return im
    else:
        return reduce(im,times-1,sigma)

def enlarge(image, pyramidDimensions, times=1):
    """
    Enlarges the given image the number of times specified using the pyramid dimensions given.
    The image is enlarged by adding a zero between every two pixels and then convolving with a
    kernel that assigns the zeros the average of the neighboring pixels and preserves the existing
    ones.

    :param image: numpy 2d array of an image.
    :param pyramidDimensions: A dictionary with keys being the dimensions of each layer in the
    pyramid and values being whether a row/column of zeros needs to be added when enlarging.
    :param times: The number of times to enlarge.
    :return: The enlarged image
    """

    if times == 0 or pyramidDimensions[image.shape] is None:
        return image

    padding = pyramidDimensions[image.shape]
    padding_x, padding_y = padding[0], padding[1]

    enlarged = np.zeros((image.shape[0]*2+padding_x,image.shape[1]*2+padding_y))
    enlarged[1::2, 1::2] = image
    kernel = np.array(([1,2,1],[2,4,2],[1,2,1])) / 4

    im = ndimage.convolve(enlarged,kernel)
    if times == 1:
        return im
    else:
        return enlarge(im, pyramidDimensions, times - 1)

