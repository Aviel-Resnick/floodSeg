import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import cv2
import sys
import numpy as np
from skimage import segmentation
import torch.nn.init
import random

args = 0

#ogImage = cv2.imread("184-6.jpg")
#segmented = cv2.imread("184-6 Output.png")
#labelColors = [[255,225,25],[0,130,200],[245,130,48],[250,190,190],[230,190,255],[128,0,0],[0,0,128],[128,128,128],[255,255,255],[0,0,0]]

# TEMP: presegmented image

#imTargetRGB = cv2.imread("blah.png")
#image = cv2.imread("175-4.jpg")
#labelColors = [[255,225,25],[0,130,200],[245,130,48],[250,190,190],[230,190,255],[128,0,0],[0,0,128],[128,128,128],[255,255,255],[0,0,0]]

#preSegGray = cv2.cvtColor(preSeg, cv2.COLOR_BGR2GRAY)

# class to pull configuration arguments from a seperate file
class LoadFromFile (argparse.Action):
    def __call__ (self, parser, namespace, values, option_string = None):
        with values as f:
            parser.parse_args(f.read().split(), namespace)

# CNN model
class CNN(nn.Module):
    def __init__(self,input_dim):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, args.numChannels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(args.numChannels)
        self.conv2 = nn.ModuleList()
        self.bn2 = nn.ModuleList()
        for i in range(args.numConv-1):
            self.conv2.append(nn.Conv2d(args.numChannels, args.numChannels, kernel_size=3, stride=1, padding=1))
            self.bn2.append(nn.BatchNorm2d(args.numChannels))
        self.conv3 = nn.Conv2d(args.numChannels, args.numChannels, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(args.numChannels)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.bn1(x)
        for i in range(args.numConv-1):
            x = self.conv2[i](x)
            x = F.relu(x)
            x = self.bn2[i](x)
        x = self.conv3(x)
        x = self.bn3(x)
        return x

def display(windowName, window, notableContours):
    for contour in notableContours:
        cv2.drawContours(window, contour, -1, (0, 0, 255), 3)
    
    cv2.imshow(windowName, window)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main(inputImage, configFile):
    global args
    cudaAvailable = torch.cuda.is_available()

    # parser setup
    parser = argparse.ArgumentParser(description='Unsupervised Arterial Segmentation ')
    parser.add_argument('--numChannels', metavar='N', default=120, type=int, help='number of channels')
    parser.add_argument('--maxIter', metavar='T', default=10, type=int, help='number of maximum iterations')
    parser.add_argument('--minLabels', metavar='minL', default=2, type=int, help='minimum number of labels')
    parser.add_argument('--lr', metavar='LR', default=0.1, type=float, help='learning rate')
    parser.add_argument('--numConv', metavar='M', default=2, type=int, help='number of convolutional layers')
    parser.add_argument('--numSuperpixels', metavar='K', default=5000, type=int, help='number of superpixels')
    parser.add_argument('--compactness', metavar='C', default=100, type=float, help='compactness of superpixels')
    parser.add_argument('--visualize', metavar='1 or 0', default=1, type=int, help='visualization flag')
    parser.add_argument('--sigma', metavar='S', default=10, type=float, help='gaussian smoothing')
    parser.add_argument('--input', metavar='FILENAME', help='input image file name', required=False)
    parser.add_argument('--minContour', metavar='minC', default=1000, type=int, help='Minimum size for a "notable" contour')
    parser.add_argument('--file', type=open, action=LoadFromFile)

    args = parser.parse_args()
    # load image
    #image = cv2.imread(inputImage)

    image = inputImage

    # scale images back down (divide by 255)
    # changes format from NHWC --> NCWH 
    data = torch.from_numpy(np.array([image.transpose((2, 0, 1)).astype('float32')/255.0]))

    # TEMP: checks cuda avaliablity
    print(cudaAvailable)

    if cudaAvailable:
        data = data.cuda()

    data = Variable(data)

    # Segments image using k-means clustering in Color-(x,y,z) space.
    labels = segmentation.slic(image, compactness=args.compactness, sigma=args.sigma, n_segments=args.numSuperpixels)
    labels = labels.reshape(image.shape[0]*image.shape[1])
    uniqueLabels = np.unique(labels)
    l_inds = []
    for i in range(len(uniqueLabels)):
        l_inds.append(np.where(labels == uniqueLabels[i])[0])

    # train
    model = CNN(data.size(1))
    if cudaAvailable:
        model.cuda()
    model.train()

    lossFunction = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9) # stochastic gradient descent 

    # generates a 100 x 3 list of integers up to 255 (100 different colors)
    # label_colours = np.random.randint(255, size=(100,3))

    labelColors = [[255,225,25],[0,130,200],[245,130,48],[250,190,190],[230,190,255],[128,0,0],[0,0,128],[128,128,128],[255,255,255],[0,0,0]]

    print(labelColors)

    # optimizes maxIter times
    for batchIndex in range(args.maxIter):
        # forwarding
        optimizer.zero_grad()
        output = model(data)[0]
        output = output.permute(1, 2, 0).contiguous().view(-1, args.numChannels)
        _, target = torch.max(output, 1)
        imTarget = target.data.cpu().numpy()
        nLabels = len(np.unique(imTarget))

        # display image
        if args.visualize:
            # colors in segments
            imTargetRGB = np.array([labelColors[c % 10] for c in imTarget])
            imTargetRGB = imTargetRGB.reshape(image.shape).astype(np.uint8)
            cv2.imshow("Result", imTargetRGB)
            cv2.waitKey(10)

        # superpixel refinement
        # TODO: use Torch Variable instead of numpy for faster calculation
        for i in range(len(l_inds)):
            labels_per_sp = imTarget[l_inds[i]]
            u_labels_per_sp = np.unique(labels_per_sp)
            hist = np.zeros(len(u_labels_per_sp))
            for j in range(len(hist)):
                hist[j] = len(np.where(labels_per_sp == u_labels_per_sp[j])[0])
            imTarget[l_inds[i]] = u_labels_per_sp[np.argmax(hist)]
        
        target = torch.from_numpy(imTarget)
        if cudaAvailable:
            target = target.cuda()

        target = Variable(target)
        loss = lossFunction(output, target)
        loss.backward()
        optimizer.step()

        # print (batch_idx, '/', args.maxIter, ':', nLabels, (loss.data[0]))
        
        # for pytorch 1.0
        # print (batch_idx, '/', args.maxIter, ':', nLabels, loss.item())
        if nLabels <= args.minLabels:
            print ("nLabels", nLabels, "reached minLabels", args.minLabels, ".")
            break

    if not args.visualize:
        output = model(data)[0]
        output = output.permute(1, 2, 0).contiguous().view(-1, args.numChannels)
        _, target = torch.max(output, 1)
        imTarget = target.data.cpu().numpy()
        imTargetRGB = np.array([labelColors[c % 10] for c in imTarget])
        imTargetRGB = imTargetRGB.reshape(image.shape).astype(np.uint8)
        
    # save output image
    outputName = str(inputImage).split('.')[0] + " Output.png"
    cv2.imwrite(outputName, imTargetRGB)

    for i in labelColors:
        imageContoured = image.copy()
        notableContours = []
    
        lowerLim = np.array(i)
        upperLim = np.array(i)
    
        mask = cv2.inRange(imTargetRGB, lowerLim, upperLim)
        returned, contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    
        colorExists = False
    
        for contour in contours:
            area = cv2.contourArea(contour)
            arc = cv2.arcLength(contour, False)
            if area > args.minContour:
                print("we have a", area, "contour")
                notableContours.append(contour)
                colorExists = True
    
        outputName = ("Contour "+str(i)+" .png")
        
        if colorExists:
            display(outputName, imageContoured, notableContours)
            print(hierarchy)

if __name__ == '__main__':
    main()

