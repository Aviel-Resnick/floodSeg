'''
Aviel Resnick, 2020
Utility designed for the automated, supervised, or manual morphometry of images, particularly stent-implanted coronary arteries.

main.py - Computations, misc. functions, logic
'''
import tkinter as tk
import tkinter.filedialog
import tkinter.simpledialog
from supervisedSeg import SelectionWindow
import numpy as np
from math import sqrt
import cv2

#finalComps = {"Lumen" : [], "Neointima" : [], "Media" : [], "Stents" : [], "Thickness" : []}
finalComps = {}

def addComponent(tree):
    global finalComps

    inputDialog = tk.simpledialog.askstring("Add new component", "Component Name:")
    finalComps.update({inputDialog : []})
    refreshTree(tree)

def removeComponent(tree):
    global finalComps
    selected = tree.focus()
    currentComponent = tree.item(selected)
    componentName = currentComponent["text"]

    print(finalComps)
    print(componentName)

    try: 
        del finalComps[componentName]
    except:
        parent = componentName.split(" ")
        index = parent.pop()
        parent = "".join(parent)
        del finalComps[parent][int(index)]

    refreshTree(tree)

def presets(tree, preset):
    global finalComps
    # TODO make it pull this from a text file
    if(preset == "Arterial Segmentation"):
        finalComps = {"Lumen" : [], "Neointima" : [], "Media" : [], "Stents" : [], "Thickness" : []}
        refreshTree(tree)

def refreshTree(tree):
    tree.delete(*tree.get_children()) # kill all the children

    for i, comp in enumerate(finalComps):
        head = tree.insert("", i, text=str(comp))
        for x, subComp in enumerate(finalComps[comp]):
            tree.insert(head, x, text=(str(comp) + " " + str(x)), values="Saved")

def getDist(pointA, pointB):
        x1 = pointA[0]
        y1 = pointA[1]
        x2 = pointB[0]
        y2 = pointB[1]

        return(sqrt((x2 - x1)**2 + (y2 - y1)**2))

def trimPoints(points):
    new = []

    for i in points:
        if not new or getDist(i, new[-1]) >= 1:
            new.append(i)

    return new

# orders points so that the contour is displayed correctly
def orderPoints(points):
    newPoints = []
    oldPoints = points.copy()

    oldPoints = trimPoints(oldPoints)

    if len(oldPoints) > 0:
        newPoints.append(oldPoints.pop())

    while len(oldPoints) > 0:
        currentPoint = newPoints[-1]
        nextPoint = None
        smallestDist = 999999999999 # lmao praying the distance between two points isn't over a trillion

        for checkPoint in oldPoints:
            dist = getDist(currentPoint, checkPoint)
            
            if dist < smallestDist:
                nextPoint = checkPoint
                smallestDist = dist

        newPoints.append(nextPoint)
        oldPoints.remove(nextPoint)

    return newPoints

def pointsToContour(points):
    a = np.array(points)
    return([a])

def saveContour(tree, contour):    
    selected = tree.focus()
    currentComponent = tree.item(selected)
    componentName = currentComponent["text"]

    if componentName != "":
        print(componentName)
        currentName = componentName.split(" ")[0]
        if len(componentName.split(" ")) > 1:
            currentId = int(componentName.split(" ")[1])
        else:
            currentId = -1
    else:
        currentId = -1

    print("Saving", componentName)

    if currentName in finalComps:
        print("ALREADY EXISTS")
        #finalComps[currentName].append(contour) # this just adds it on the end, we want to add it to a particular spot (ID)
        if currentId != -1:
            finalComps[currentName][currentId] = contour
        else:
            finalComps[currentName].append(contour)

    else:
        finalComps.update({currentName : [contour]})

    refreshTree(tree)

def floodFill(image, tree):
    repeat = True
    
    while repeat:
        selection = SelectionWindow('Selection Window', image, connectivity=8)
        output = selection.show(verbose=True)
        repeat = output[0]
        returnedContours = output[1]

        exteriorContour = None
        largestArea = 0
        
        for i in returnedContours:
            currentPoints = []
            for x in i:
                xVal = x[0][0]
                yVal = x[0][1]
                currentPoints.append((xVal,yVal))
            
            points = orderPoints(currentPoints)
            hull = cv2.convexHull(pointsToContour(points)[0])
            area = cv2.contourArea(pointsToContour(hull)[0])
            
            if(area > largestArea):
                largestArea = area
                exteriorContour = pointsToContour(hull)[0] # should be hull

        #self.completeContour(exteriorContour)
        saveContour(tree, pointsToContour(exteriorContour)) # why am I ptc'ing exterior contour twice
        print("Saved Contour")

def togglePreview(tree):
    selected = tree.focus()
    currentComponent = tree.item(selected)
    componentName = currentComponent["text"]

    compParent = componentName.split(" ")[0]
    if len(componentName.split(" ")) > 1:
        compId = componentName.split(" ")[1]

    if compParent in finalComps:
        if len(componentName.split(" ")) == 1: # don't think this exists anymore, since even the first contour is SEG 0
            img = cv2.imread(pathToImg)
            for i in finalComps[compParent]:
                cv2.drawContours(img, i, 0, (0,0,255), 2)

            cv2.imshow("Viewing Contour", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            img = cv2.imread(pathToImg)
            contour = (finalComps[compParent])[int(compId)]
            cv2.drawContours(img, contour, 0, (0,0,255), 2)

            cv2.imshow("Viewing Contour", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    