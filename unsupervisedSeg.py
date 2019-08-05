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

cudaAvailable = torch.cuda.is_available()

# class to pull configuration arguments from a seperate file
class LoadFromFile (argparse.Action):
    def __call__ (self, parser, namespace, values, option_string = None):
        with values as f:
            parser.parse_args(f.read().split(), namespace)

# parser setup
parser = argparse.ArgumentParser(description='Unsupervised Arterial Segmentation ')
parser.add_argument('--numChannels', metavar='N', default=100, type=int, help='number of channels')
parser.add_argument('--maxIter', metavar='T', default=1000, type=int, help='number of maximum iterations')
parser.add_argument('--minLabels', metavar='minL', default=5, type=int, help='minimum number of labels')
parser.add_argument('--lr', metavar='LR', default=0.1, type=float, help='learning rate')
parser.add_argument('--numConv', metavar='M', default=2, type=int, help='number of convolutional layers')
parser.add_argument('--numSuperpixels', metavar='K', default=10000, type=int, help='number of superpixels')
parser.add_argument('--compactness', metavar='C', default=100, type=float, help='compactness of superpixels')
parser.add_argument('--visualize', metavar='1 or 0', default=1, type=int, help='visualization flag')
parser.add_argument('--input', metavar='FILENAME', help='input image file name', required=False)
parser.add_argument('--file', type=open, action=LoadFromFile)
args = parser.parse_args()

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

# load image
image = cv2.imread(args.input)

# scale images back down (divide by 255)
# changes format from NHWC --> NCWH 
data = torch.from_numpy(np.array([image.transpose((2, 0, 1)).astype('float32')/255.0]))

# TEMP: checks cuda avaliablity
print(cudaAvailable)

if cudaAvailable:
    data = data.cuda()

data = Variable(data)

# Segments image using k-means clustering in Color-(x,y,z) space.
labels = segmentation.slic(image, compactness=args.compactness, n_segments=args.numSuperpixels)
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
label_colours = np.random.randint(255, size=(100,3))
print(label_colours)

# optimizes maxIter times
for batchIndex in range(args.maxIter):
    # forwarding
    optimizer.zero_grad()
    output = model(data)[0]
    output = output.permute(1, 2, 0).contiguous().view(-1, args.numChannels)
    _, target = torch.max(output, 1)
    im_target = target.data.cpu().numpy()
    nLabels = len(np.unique(im_target))

    # display image
    if args.visualize:
        # colors in segments
        im_target_rgb = np.array([label_colours[c % 100] for c in im_target])
        im_target_rgb = im_target_rgb.reshape(image.shape).astype(np.uint8)
        cv2.imshow("Result", im_target_rgb)
        cv2.waitKey(10)

    # superpixel refinement
    # TODO: use Torch Variable instead of numpy for faster calculation
    for i in range(len(l_inds)):
        labels_per_sp = im_target[l_inds[i]]
        u_labels_per_sp = np.unique(labels_per_sp)
        hist = np.zeros(len(u_labels_per_sp))
        for j in range(len(hist)):
            hist[j] = len(np.where(labels_per_sp == u_labels_per_sp[j])[0])
        im_target[l_inds[i]] = u_labels_per_sp[np.argmax(hist)]
    
    target = torch.from_numpy(im_target)
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
    im_target = target.data.cpu().numpy()
    im_target_rgb = np.array([label_colours[c % 100] for c in im_target])
    im_target_rgb = im_target_rgb.reshape(image.shape).astype(np.uint8)
    
# save output image
outputName = str(args.input).split('.')[0] + " Output.png"
cv2.imwrite(outputName, im_target_rgb)
