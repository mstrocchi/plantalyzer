import cv2,time,argparse,glob
import numpy as np

#global variable to keep track of 
show = False
reference_image = cv2.imread("resources/dataset/tags/001.jpeg")


def onTrackbarActivity(x):
    global show
    show = True
    pass


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


if __name__ == '__main__' :

    # Get the filename from the command line 
    files = glob.glob('resources/dataset/sick_leaves/0*.jpeg')
    files.sort()
    # load the image 
    original = cv2.imread(files[0])
    #Resize the image
    rsize = 250
    original = cv2.resize(original,(rsize,rsize))
    original = color_transfer(original, reference_image)

    #position on the screen where the windows start
    initialX = 50
    initialY = 50

    #creating windows to display images
    cv2.namedWindow('P-> Previous, N-> Next',cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow('SelectBGR',cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow('SelectHSV',cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow('SelectYCB',cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow('SelectLAB',cv2.WINDOW_AUTOSIZE)

    #moving the windows to stack them horizontally
    cv2.moveWindow('P-> Previous, N-> Next',initialX,initialY)
    cv2.moveWindow('SelectBGR',initialX + (rsize + 5),initialY)
    cv2.moveWindow('SelectHSV',initialX + 2*(rsize + 5),initialY)
    cv2.moveWindow('SelectYCB',initialX + 3*(rsize + 5),initialY)
    cv2.moveWindow('SelectLAB',initialX + 4*(rsize + 5),initialY)

    #creating trackbars to get values for YCrCb
    cv2.createTrackbar('CrMin','SelectYCB',0,255,onTrackbarActivity)
    cv2.createTrackbar('CrMax','SelectYCB',0,255,onTrackbarActivity)
    cv2.createTrackbar('CbMin','SelectYCB',0,255,onTrackbarActivity)
    cv2.createTrackbar('CbMax','SelectYCB',0,255,onTrackbarActivity)
    cv2.createTrackbar('YMin','SelectYCB',0,255,onTrackbarActivity)
    cv2.createTrackbar('YMax','SelectYCB',0,255,onTrackbarActivity)

    #creating trackbars to get values for HSV
    cv2.createTrackbar('HMin','SelectHSV',0,180,onTrackbarActivity)
    cv2.createTrackbar('HMax','SelectHSV',0,180,onTrackbarActivity)
    cv2.createTrackbar('SMin','SelectHSV',0,255,onTrackbarActivity)
    cv2.createTrackbar('SMax','SelectHSV',0,255,onTrackbarActivity)
    cv2.createTrackbar('VMin','SelectHSV',0,255,onTrackbarActivity)
    cv2.createTrackbar('VMax','SelectHSV',0,255,onTrackbarActivity)

    #creating trackbars to get values for BGR
    cv2.createTrackbar('BGRBMin','SelectBGR',0,255,onTrackbarActivity)
    cv2.createTrackbar('BGRBMax','SelectBGR',0,255,onTrackbarActivity)
    cv2.createTrackbar('BGRGMin','SelectBGR',0,255,onTrackbarActivity)
    cv2.createTrackbar('BGRGMax','SelectBGR',0,255,onTrackbarActivity)
    cv2.createTrackbar('BGRRMin','SelectBGR',0,255,onTrackbarActivity)
    cv2.createTrackbar('BGRRMax','SelectBGR',0,255,onTrackbarActivity)

    #creating trackbars to get values for LAB
    cv2.createTrackbar('LABLMin','SelectLAB',0,255,onTrackbarActivity)
    cv2.createTrackbar('LABLMax','SelectLAB',0,255,onTrackbarActivity)
    cv2.createTrackbar('LABAMin','SelectLAB',0,255,onTrackbarActivity)
    cv2.createTrackbar('LABAMax','SelectLAB',0,255,onTrackbarActivity)
    cv2.createTrackbar('LABBMin','SelectLAB',0,255,onTrackbarActivity)
    cv2.createTrackbar('LABBMax','SelectLAB',0,255,onTrackbarActivity)

    # show all images initially
    cv2.imshow('SelectHSV',original)
    cv2.imshow('SelectYCB',original)
    cv2.imshow('SelectLAB',original)
    cv2.imshow('SelectBGR',original)
    i = 0
    while(1):

        cv2.imshow('P-> Previous, N-> Next',original)  
        k = cv2.waitKey(1) & 0xFF

        # check next image in folder    
        if k == ord('n'):
            i += 1
            original = cv2.imread(files[i%len(files)])
            original = cv2.resize(original,(rsize,rsize))
            show = True
 
        # check previous image in folder    
        elif k == ord('p'):
            i -= 1
            original = cv2.imread(files[i%len(files)])
            original = cv2.resize(original,(rsize,rsize))
            show = True
        # Close all windows when 'esc' key is pressed
        elif k == 27:
            break
        
        if show: # If there is any event on the trackbar
            show = False

            # Get values from the BGR trackbar
            BMin = cv2.getTrackbarPos('BGRBMin','SelectBGR')
            GMin = cv2.getTrackbarPos('BGRGMin','SelectBGR')
            RMin = cv2.getTrackbarPos('BGRRMin','SelectBGR')
            BMax = cv2.getTrackbarPos('BGRBMax','SelectBGR')
            GMax = cv2.getTrackbarPos('BGRGMax','SelectBGR')
            RMax = cv2.getTrackbarPos('BGRRMax','SelectBGR')
            minBGR = np.array([BMin, GMin, RMin])
            maxBGR = np.array([BMax, GMax, RMax])

            # Get values from the HSV trackbar
            HMin = cv2.getTrackbarPos('HMin','SelectHSV')
            SMin = cv2.getTrackbarPos('SMin','SelectHSV')
            VMin = cv2.getTrackbarPos('VMin','SelectHSV')
            HMax = cv2.getTrackbarPos('HMax','SelectHSV')
            SMax = cv2.getTrackbarPos('SMax','SelectHSV')
            VMax = cv2.getTrackbarPos('VMax','SelectHSV')
            minHSV = np.array([HMin, SMin, VMin])
            maxHSV = np.array([HMax, SMax, VMax])

            # Get values from the LAB trackbar
            LMin = cv2.getTrackbarPos('LABLMin','SelectLAB')
            AMin = cv2.getTrackbarPos('LABAMin','SelectLAB')
            BMin = cv2.getTrackbarPos('LABBMin','SelectLAB')
            LMax = cv2.getTrackbarPos('LABLMax','SelectLAB')
            AMax = cv2.getTrackbarPos('LABAMax','SelectLAB')
            BMax = cv2.getTrackbarPos('LABBMax','SelectLAB')
            minLAB = np.array([LMin, AMin, BMin])
            maxLAB = np.array([LMax, AMax, BMax])

            # Get values from the YCrCb trackbar
            YMin = cv2.getTrackbarPos('YMin','SelectYCB')
            CrMin = cv2.getTrackbarPos('CrMin','SelectYCB')
            CbMin = cv2.getTrackbarPos('CbMin','SelectYCB')
            YMax = cv2.getTrackbarPos('YMax','SelectYCB')
            CrMax = cv2.getTrackbarPos('CrMax','SelectYCB')
            CbMax = cv2.getTrackbarPos('CbMax','SelectYCB')
            minYCB = np.array([YMin, CrMin, CbMin])
            maxYCB = np.array([YMax, CrMax, CbMax])
            
            # Convert the BGR image to other color spaces
            imageBGR = np.copy(original)
            imageHSV = cv2.cvtColor(original,cv2.COLOR_BGR2HSV)
            imageYCB = cv2.cvtColor(original,cv2.COLOR_BGR2YCrCb)
            imageLAB = cv2.cvtColor(original,cv2.COLOR_BGR2LAB)

            # Create the mask using the min and max values obtained from trackbar and apply bitwise and operation to get the results         
            maskBGR = cv2.inRange(imageBGR,minBGR,maxBGR)
            resultBGR = cv2.bitwise_and(original, original, mask = maskBGR)         
            
            maskHSV = cv2.inRange(imageHSV,minHSV,maxHSV)
            resultHSV = cv2.bitwise_and(original, original, mask = maskHSV)
            
            maskYCB = cv2.inRange(imageYCB,minYCB,maxYCB)
            resultYCB = cv2.bitwise_and(original, original, mask = maskYCB)         
        
            maskLAB = cv2.inRange(imageLAB,minLAB,maxLAB)
            resultLAB = cv2.bitwise_and(original, original, mask = maskLAB)         
            
            # Show the results
            cv2.imshow('SelectBGR',resultBGR)
            cv2.imshow('SelectYCB',resultYCB)
            cv2.imshow('SelectLAB',resultLAB)
            cv2.imshow('SelectHSV',resultHSV)

    cv2.destroyAllWindows()

