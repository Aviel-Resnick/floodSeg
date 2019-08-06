import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import skimage.data as data
import skimage.segmentation as seg
import skimage.filters as filters
import skimage.draw as draw
import skimage.color as color
from skimage import io
from skimage import data
from skimage.io import imread

def mediaContours(img):
    im = cv2.imread(img) # main layer
    outerIm = cv2.imread(img) # single outer contour layer

    blurredIm = cv2.GaussianBlur(im, (5,5), 0)
    hsv = cv2.cvtColor(blurredIm, cv2.COLOR_BGR2HSV)

    lowerLim = np.array([0,0,0])
    upperLim = np.array([177,156,159])

    mask = cv2.inRange(hsv, lowerLim, upperLim)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    media = []
    eel = 0

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 20000:
            media.append(contour)
            cv2.drawContours(im, contour, -1, (0, 255, 0), 3)
            if len(media) == 1:
                eel = contour
        
    cv2.drawContours(im, eel, -1, (0, 0, 255), 3)
    cv2.imshow("Frame", im)
        #cv2.imshow("Mask", mask)

    print("Media Area: " + str(cv2.contourArea(media[0]) - cv2.contourArea(media[1])))
    print("EEL: " + str(cv2.arcLength(media[0], True)))

    while True:
        key = cv2.waitKey(1)
        if key == 27:
            break

    cv2.destroyAllWindows()
    
def image_show(image, nrows=1, ncols=1, cmap='gray'):
    ax.plot(points[:, 0], points[:, 1], '--r', lw=3)
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 14))
    ax.imshow(image, cmap='gray')
    ax.axis('off')
    io.show()
    return fig, ax

def seg(imgPath):
    image = imread(imgPath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_gray = color.rgb2gray(image) 
    #img_threshold = filters.threshold_local(image, block_size=51, offset=10) 
    image_show(image_gray);

def circle_points(resolution, center, radius):
    radians = np.linspace(0, 2*np.pi, resolution)
    c = center[1] + radius*np.cos(radians)#polar co-ordinates
    r = center[0] + radius*np.sin(radians)
    
    return np.array([c, r]).T

def snake(imgPath):
    image = imread(imgPath)
    image_gray = color.rgb2gray(image)
    points = circle_points(200, [80, 250], 80)[:-1]

def gaborFilter(img):
    g_kernel = cv2.getGaborKernel((21, 21), 8.0, np.pi/4, 10.0, 0.5, 0, ktype=cv2.CV_32F)

    img = cv2.imread(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    filtered_img = cv2.filter2D(img, cv2.CV_8UC3, g_kernel)

    cv2.imshow('image', img)
    cv2.imshow('filtered image', filtered_img)

    h, w = g_kernel.shape[:2]
    g_kernel = cv2.resize(filtered_img, (3*w, 3*h), interpolation=cv2.INTER_CUBIC)
    cv2.imshow('gabor kernel (resized)', g_kernel)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def unitConversion(calibration, pxValue): # pixel to metric conversion
    # calibration is in the form 1 mm to how many pixels
    return pxValue / calibration

filenames = glob.glob("img/*.jpg")

if __name__ == "__main__":
    for imgPath in filenames:
        print(imgPath)
        mediaContours(imgPath)