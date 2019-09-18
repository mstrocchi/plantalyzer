import cv2
import glob
import copy
import numpy as np


imageScaleFactor = 45
reference_image = cv2.imread("resources/4K/002.jpeg")
dir = 'resources/FinalTest/0*.jpeg'

# Overall filter values to isolate for sick leaves + FPs
LABmin = np.array([68, 123, 138], np.uint8)
LABmax = np.array([255, 162, 255], np.uint8)

# Range filter values for healthy leaves
LABmin_healthy = np.array([0, 83, 124], np.uint8)
LABmax_healthy = np.array([255, 129, 188], np.uint8)
# LABmax_healthy = np.array([192, 124, 171], np.uint8)

# Range filter values for terrain
LABmin_terrain = np.array([0, 129, 0], np.uint8)
LABmax_terrain = np.array([255, 255, 148], np.uint8)

# Range filter values for yellow leaves and tags (FPs)
HSVmin_yellow = np.array([19, 80, 174], np.uint8)
HSVmax_yellow = np.array([33, 255, 255], np.uint8)


def stackingWindows():
    space = 50
    offset = 70
    cv2.moveWindow("Original image", space, space)
    cv2.moveWindow("Keypoints original", space, hsize + space + offset) #space, hsize + space + offset
    cv2.moveWindow("Color matched", wsize + space, space)
    cv2.moveWindow("Keypoints Dark", wsize + space, hsize + space + offset) #wsize + space, hsize + space + offset)

    return


def filterInRange(frame, min, max, colorMode):
    """
    Filters the pixel that are in the specified color
    range, following the specified colorMode.

    :param frame: BGR image.
    :param min: min color val.
    :param max: max color val.
    :param colorMode: Color space conversion
    :return: filtered frame in BGR
    returns image with pixels NOT in range
    """

    tempFrame = cv2.cvtColor(frame, colorMode)

    mask = cv2.inRange(tempFrame, min, max)
    mask = cv2.bitwise_not(mask)

    filtered_frame = cv2.bitwise_and(frame, frame, mask=mask)

    return filtered_frame


def filterNotInRange(frame, min, max, colorMode):
    """
    Filters the pixel that are NOT in the specified color
    range, following the specified colorMode.

    :param frame: BGR image.
    :param min: min color val.
    :param max: max color val.
    :param colorMode: Color space conversion
    :return: filtered frame in BGR
    returns image with pixels in range
    """

    tempFrame = cv2.cvtColor(frame, colorMode)

    mask = cv2.inRange(tempFrame, min, max)

    filtered_frame = cv2.bitwise_and(frame, frame, mask=mask)

    return filtered_frame


def histogram_equalize(img):
    """
    DEPRECATED
    Takes the input image and equalizes the histogram on HSV
    color space.

    NB: The YUV color space might not be the best color space to deal with.

    :param img: BGR image.
    :return: histogram equalized BGR image.
    """

    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    # equalize the histogram in all the YUV channels
    # for c in range(0, 2):
    img_yuv[:, :, 2] = cv2.equalizeHist(img_yuv[:, :, 2])
    img_yuv[:, :, 1] = cv2.equalizeHist(img_yuv[:, :, 1])
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])

    # convert the YUV image back to RGB format
    img_output = cv2.cvtColor(img_yuv, cv2.COLOR_LAB2BGR)

    return img_output


# dilatation followed by erosion (fills small gaps)
def closing(img, kernel):
    return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)


# erosion followed by dilatation (deletes spots on background)
def opening(img, kernel):
    return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)


def differentialNode(input, filter):
    return cv2.subtract(input, filter)


def filteringEngine(original, debug=True):

    processedImage1 = filterNotInRange(original, LABmin_healthy, LABmax_healthy, cv2.COLOR_BGR2LAB)
    processedImage2 = filterNotInRange(original, LABmin_terrain, LABmax_terrain, cv2.COLOR_BGR2LAB)
    # Image containing many FPs
    processedImage3 = filterNotInRange(original, HSVmin_yellow, HSVmax_yellow, cv2.COLOR_BGR2HSV)

    sum1 = cv2.add(processedImage1, processedImage2)
    sub1 = differentialNode(original, sum1)

    processedImage = filterNotInRange(sub1, LABmin, LABmax, cv2.COLOR_BGR2LAB)
    # sum2 = cv2.add(processedImage, processedImage3)

    kernel = np.ones((6, 6), np.uint8)
    temp = closing(processedImage, kernel)

    kernel = np.ones((3, 3), np.uint8)
    out = opening(temp, kernel)

    if debug:
        # cv2.imshow('processedImage1', processedImage1)
        # cv2.imshow('processedImage2', processedImage2)
        cv2.imshow('processedImage3', processedImage3)
        # cv2.imshow('sum1', sum1)
        # cv2.imshow('sub1', sub1)
        # cv2.imshow('processedImage', processedImage)
        # cv2.imshow('sum2', sum2)
        # cv2.imshow('out', out)

    return out


def blob_detector(filtered_frame, original_frame):

    # create a bi-color image.
    hsv = cv2.cvtColor(filtered_frame, cv2.COLOR_BGR2HSV)
    _, saturation, _ = cv2.split(hsv)
    _, thresholded = cv2.threshold(saturation, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresholded = cv2.bitwise_not(thresholded)

    # Setup SimpleBlobDetector parameters.
    params = cv2.SimpleBlobDetector_Params()

    params.filterByConvexity = False
    params.filterByInertia = False
    params.filterByArea = False
    # params.minArea = 12

    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(thresholded)

    # Draw detected blobs as red circles.
    keypointsOriginal = cv2.drawKeypoints(original_frame, keypoints, np.array([]), (0, 0, 255),
                                          cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    keypointsDark = cv2.drawKeypoints(filtered_frame, keypoints, np.array([]), (0, 0, 255),
                                            cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    return keypointsOriginal, keypointsDark


def contour_blob(filtered_frame, original_frame, blob_thershold, debug=False):

    hsv = cv2.cvtColor(filtered_frame, cv2.COLOR_BGR2HSV)
    _, saturation, _ = cv2.split(hsv)

    _, thresholded = cv2.threshold(saturation, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(thresholded, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contour_list = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > blob_thershold:
            contour_list.append(contour)

    cv2.drawContours(original_frame, contour_list, -1, (255, 0, 255), 5)

    if debug:
        cv2.imshow('Saturation Image', saturation)
        cv2.imshow('Thresholded Image', thresholded)
        cv2.imshow('Objects Detected', original_frame)


def color_transfer(source, target, clip=True, preserve_paper=True):
    """
    Transfers the color distribution from the source to the target
    image using the mean and standard deviations of the L*a*b*
    color space.
    This implementation is (loosely) based on to the "Color Transfer
    between Images" paper by Reinhard et al., 2001.
    Parameters:
    -------
    source: NumPy array
        OpenCV image in BGR color space (the source image)
    target: NumPy array
        OpenCV image in BGR color space (the target image)
    clip: Should components of L*a*b* image be scaled by np.clip before
        converting back to BGR color space?
        If False then components will be min-max scaled appropriately.
        Clipping will keep target image brightness truer to the input.
        Scaling will adjust image brightness to avoid washed out portions
        in the resulting color transfer that can be caused by clipping.
    preserve_paper: Should color transfer strictly follow methodology
        layed out in original paper? The method does not always produce
        aesthetically pleasing results.
        If False then L*a*b* components will scaled using the reciprocal of
        the scaling factor proposed in the paper.  This method seems to produce
        more consistently aesthetically pleasing results
    Returns:
    -------
    transfer: NumPy array
        OpenCV image (w, h, 3) NumPy array (uint8)
    """
    # convert the images from the RGB to L*ab* color space, being
    # sure to utilizing the floating point data type (note: OpenCV
    # expects floats to be 32-bit, so use that instead of 64-bit)
    source = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype("float32")
    target = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype("float32")

    # compute color statistics for the source and target images
    (lMeanSrc, lStdSrc, aMeanSrc, aStdSrc, bMeanSrc, bStdSrc) = image_stats(source)
    (lMeanTar, lStdTar, aMeanTar, aStdTar, bMeanTar, bStdTar) = image_stats(target)

    # subtract the means from the target image
    (l, a, b) = cv2.split(target)
    l -= lMeanTar
    a -= aMeanTar
    b -= bMeanTar

    if preserve_paper:
        # scale by the standard deviations using paper proposed factor
        l = (lStdTar / lStdSrc) * l
        a = (aStdTar / aStdSrc) * a
        b = (bStdTar / bStdSrc) * b
    else:
        # scale by the standard deviations using reciprocal of paper proposed factor
        l = (lStdSrc / lStdTar) * l
        a = (aStdSrc / aStdTar) * a
        b = (bStdSrc / bStdTar) * b

    # add in the source mean
    l += lMeanSrc
    a += aMeanSrc
    b += bMeanSrc

    # clip/scale the pixel intensities to [0, 255] if they fall
    # outside this range
    l = _scale_array(l, clip=clip)
    a = _scale_array(a, clip=clip)
    b = _scale_array(b, clip=clip)

    # merge the channels together and convert back to the RGB color
    # space, being sure to utilize the 8-bit unsigned integer data
    # type
    transfer = cv2.merge([l, a, b])
    transfer = cv2.cvtColor(transfer.astype("uint8"), cv2.COLOR_LAB2BGR)

    # return the color transferred image
    return transfer


def image_stats(image):
    """
    Parameters:
    -------
    image: NumPy array
        OpenCV image in L*a*b* color space
    Returns:
    -------
    Tuple of mean and standard deviations for the L*, a*, and b*
    channels, respectively
    """
    # compute the mean and standard deviation of each channel
    (l, a, b) = cv2.split(image)
    (lMean, lStd) = (l.mean(), l.std())
    (aMean, aStd) = (a.mean(), a.std())
    (bMean, bStd) = (b.mean(), b.std())

    # return the color statistics
    return (lMean, lStd, aMean, aStd, bMean, bStd)


def _min_max_scale(arr, new_range=(0, 255)):
    """
    Perform min-max scaling to a NumPy array
    Parameters:
    -------
    arr: NumPy array to be scaled to [new_min, new_max] range
    new_range: tuple of form (min, max) specifying range of
        transformed array
    Returns:
    -------
    NumPy array that has been scaled to be in
    [new_range[0], new_range[1]] range
    """
    # get array's current min and max
    mn = arr.min()
    mx = arr.max()

    # check if scaling needs to be done to be in new_range
    if mn < new_range[0] or mx > new_range[1]:
        # perform min-max scaling
        scaled = (new_range[1] - new_range[0]) * (arr - mn) / (mx - mn) + new_range[0]
    else:
        # return array if already in range
        scaled = arr

    return scaled


def _scale_array(arr, clip=True):
    """
    Trim NumPy array values to be in [0, 255] range with option of
    clipping or scaling.
    Parameters:
    -------
    arr: array to be trimmed to [0, 255] range
    clip: should array be scaled by np.clip? if False then input
        array will be min-max scaled to range
        [max([arr.min(), 0]), min([arr.max(), 255])]
    Returns:
    -------
    NumPy array that has been scaled to be in [0, 255] range
    """
    if clip:
        scaled = np.clip(arr, 0, 255)
    else:
        scale_range = (max([arr.min(), 0]), min([arr.max(), 255]))
        scaled = _min_max_scale(arr, new_range=scale_range)

    return scaled


if __name__ == '__main__':

    # Get the filename from the command line
    files = glob.glob(dir)
    files.sort()

    # load the image
    original = cv2.imread(files[0])

    # Resize the image (16:9)
    hsize = 9 * imageScaleFactor
    wsize = 16 * imageScaleFactor
    original = cv2.resize(original, (wsize, hsize))

    i = 0
    while 1:

        color_corrected = color_transfer(reference_image, original)
        processedImage = filteringEngine(color_corrected)

        # deep copies of Original image and and processedImage
        temp = copy.deepcopy(original)
        temp1 = copy.deepcopy(processedImage)

        keypointsOriginal, keypointsDark = blob_detector(temp1, original)

        cv2.imshow('Original image', original)
        cv2.imshow('Keypoints original', keypointsOriginal)
        cv2.imshow("Color matched", color_corrected)
        cv2.imshow("Keypoints Dark", keypointsDark)
        stackingWindows()

        k = cv2.waitKey(1) & 0xFF

        # check next image in folder - update "original" image
        if k == ord('n'):
            i += 1
            original = cv2.imread(files[i % len(files)])
            original = cv2.resize(original, (wsize, hsize))

        # check previous image in folder  - update "original" image
        elif k == ord('p'):
            i -= 1
            original = cv2.imread(files[i % len(files)])
            original = cv2.resize(original, (wsize, hsize))

        # Close all windows when 'esc' key is pressed
        elif k == 27:
            break

    cv2.destroyAllWindows()
