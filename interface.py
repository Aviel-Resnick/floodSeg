'''
Aviel Resnick, 2019
Utility designed for the automated, supervised, or manual morphometry of images, particularly stent-implanted coronary arteries.

interface.py - main GUI, manual editing, misc. functions
'''

import tkinter as tk
import tkinter.filedialog
import tkinter.simpledialog
from tkinter import ttk
from PIL import ImageTk, Image, ImageEnhance, ImageDraw
from tkinter import Tk, Text, BOTH, W, N, E, S
from tkinter.ttk import Frame, Button, Label, Style
import os.path
from os import path
import webbrowser
### import unsupervisedSeg
from supervisedSeg import SelectionWindow
import cv2
import numpy as np
from math import sqrt
import xlsxwriter
from shapely.geometry import Polygon, LineString
import math
import matplotlib.pyplot as plt

COLOR = {
    True:  '#6699cc',
    False: '#ffcc33'
    }

img = None # original
pathToImg = None
currentImage = None
newImagePath = None
configFile = None
points = []
pointCount = 0
conversion = 0
componentList = [["Lumen", "Empty", None], ["Neointima", "Empty", None], ["Media", "Empty", None], ["Stents", "Empty", None], ["Thickness", "Empty", None]]
stentComponents = [["Stent 1", "Empty", None], ["Stent 2", "Empty", None], ["Stent 3", "Empty", None], ["Stent 4", "Empty", None], ["Stent 5", "Empty", None], ["Stent 6", "Empty", None], ["Stent 7", "Empty", None], ["Stent 8", "Empty", None], ["Stent 9", "Empty", None], ["Stent 10", "Empty", None]]
finalComps = {"Lumen" : [], "Neointima" : [], "Media" : [], "Stents" : [], "Thickness" : []}
finalVals = {}
currentName = ""

'''
Known bugs:
    Could crash if a file isn't selected, or no new component name is given
    Since I am not removing the "deleted" item from componentList, its returns at the next refresh
    Adding components window might be on the wrong layer

To Do:
    Excel Output
    Additive (Stent) Selection
    Comment sections
    break Segmentation up into multiple classes
    Find alternative to globals in interface
        Lambda is the answer

Notes:
    Stents often included in neointima component
        Maybe identify those independently and subtract
'''

class Segmentation(Frame):
    def __init__(self):
        super().__init__()
        self.initUI()

    def selectImage(self):
        global pathToImg, img, finalComps
        
        pathToImg = tkinter.filedialog.askopenfilename(filetypes=[("Image File",'.jpg')])
        img = Image.open(pathToImg)
        photoImg = ImageTk.PhotoImage(img)

        width, height = img.size
        
        '''
        resX = width + 150 if width + 150 > 350 else 350
        resY = height + 20 if height + 20 > 390 else 390
        #self.master.geometry(str(resX)+"x"+str(resY))
        '''

        if width > 800 or height > 800:
            basewidth = 600
            wpercent = (basewidth/float(img.size[0]))
            hsize = int((float(img.size[1])*float(wpercent)))
            img = img.resize((basewidth,hsize), Image.ANTIALIAS)
            photoImg = ImageTk.PhotoImage(img)
            
            resizedImagePath = pathToImg.split('.')[0] + " Resized.png"
            img.save(resizedImagePath)
            pathToImg = resizedImagePath

        self.adjust(1.0, 1.0, 1.0)

        # deprecated
        for i in componentList:
            i[1] = "Empty"
            i[2] = None

        finalComps = {"Lumen" : [], "Neointima" : [], "Media" : [], "Stents" : [], "Thickness" : []}

        self.updateImage(photoImg)
        
        self.master.geometry("")

    def updateImage(self, image):
        global currentImage

        panel = tk.Label(self, image = image)
        panel.image = image
        currentImage = image
        panel.grid(row=1, column=0, columnspan=2, rowspan=5, padx=5, sticky=E+W+S+N)
    
    def chooseAdjust(self):
        self.pack(fill=BOTH, expand=True)
        popup = tk.Tk()
        popup.wm_title("Image Adjustment")

        labelSharpness = tk.Label(popup, text="Sharpness:", font=("Helvetica 12 bold"))
        labelSharpness.grid(row=1, column=0, padx=5, pady=16, sticky=E+W+S+N)

        slideSharpness = tk.Scale(popup, orient='horizontal', from_=0, to=2)
        slideSharpness.set(1)
        slideSharpness.grid(row=1, column=1, padx=10, pady=0, sticky=E+W+S+N)

        labelSharpnessInfo = tk.Label(popup, text="0 = Blur, 1 = Original, 2 = Sharp")
        labelSharpnessInfo.grid(row=2, column=1, padx=10, pady=0, sticky=E+W+S+N)

        labelContrast = tk.Label(popup, text="Contrast:", font=("Helvetica 12 bold"))
        labelContrast.grid(row=3, column=0, padx=5, pady=16, sticky=E+W+S+N)

        slideContrast = tk.Scale(popup, orient='horizontal', from_=0.0, to=10.0, digits = 3, resolution = 0.1)
        slideContrast.set(1)
        slideContrast.grid(row=3, column=1, padx=10, pady=0, sticky=E+W+S+N)

        labelContrastInfo = tk.Label(popup, text="0 = Gray, 1 = Original, n > 1 = Contrast Factor")
        labelContrastInfo.grid(row=4, column=1, padx=10, pady=0, sticky=E+W+S+N)

        labelBrightness = tk.Label(popup, text="Brightness:", font=("Helvetica 12 bold"))
        labelBrightness.grid(row=5, column=0, padx=5, pady=16, sticky=E+W+S+N)

        slideBrightness = tk.Scale(popup, orient='horizontal', from_=0.0, to=10.0, digits = 3, resolution = 0.1)
        slideBrightness.set(1)
        slideBrightness.grid(row=5, column=1, padx=10, pady=0, sticky=E+W+S+N)

        labelBrightnessInfo = tk.Label(popup, text="0 = Black, 1 = Original, n > 1 = Brightness Factor")
        labelBrightnessInfo.grid(row=6, column=1, padx=10, pady=0, sticky=E+W+S+N)

        restore = tk.Button(popup, text="Restore to Original", command = lambda:[self.adjust(1, 1, 1), popup.destroy()])
        restore.grid(row=7, column=0, padx=10, pady=10, sticky=E+W+S+N)

        submit = tk.Button(popup, text="Confirm Adjustments", command = lambda:[self.adjust(slideSharpness.get(), slideContrast.get(), slideBrightness.get()), popup.destroy()])
        submit.grid(row=7, column=1, padx=10, pady=10, sticky=E+W+S+N)

        popup.mainloop()

    def adjust(self, sharp, contrast, bright):
        global img, pathToImg, newImagePath

        enhancer = ImageEnhance.Sharpness(img)
        newImage = enhancer.enhance(sharp)

        enhancer = ImageEnhance.Contrast(newImage)
        newImage = enhancer.enhance(contrast)

        enhancer = ImageEnhance.Brightness(newImage)
        newImage = enhancer.enhance(bright)
        
        #img = newImage

        # sharpness, contrast, brightness
        print("adjusting by ", sharp, contrast, bright)
        self.updateImage(ImageTk.PhotoImage(newImage))

        newImagePath = pathToImg.split('.')[0] + " Modified.png"
        newImage.save(newImagePath)

        if sharp == -1 and contrast == -1 and bright == -1:
            self.updateImage(ImageTk.PhotoImage(img))


    def tune(self):
        global configFile
        configFile = "config.txt" # default path

        # TODO: Parse text to prevent unwanted commands from executing
        while not path.exists(configFile):
            print("we don't have a config file")
            configFile = tkinter.simpledialog.askstring("Config Path", "Path to config.txt")

        webbrowser.open(configFile)

    def calibration(self):
        global conversion
        conversion = tkinter.simpledialog.askstring("Calibration", "1mm = ?px")

    def outputLoc(self):
        file = tkinter.filedialog.asksaveasfilename(confirmoverwrite=False)
        self.excelOutput(file)
     
    def segment(self):
        global pathToImg, newImagePath

        if newImagePath is None:
            newImagePath = pathToImg

        ### unsupervisedSeg.main(newImagePath, pathToImg, configFile)

    def floodFill(self, tree):
        global pathToImg, newImagePath

        if newImagePath is None:
            newImagePath = pathToImg

        image = cv2.imread(newImagePath)

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
                
                points = self.orderPoints(currentPoints)
                hull = cv2.convexHull(self.pointsToContour(points)[0])
                area = cv2.contourArea(self.pointsToContour(hull)[0])
                
                if(area > largestArea):
                    largestArea = area
                    exteriorContour = self.pointsToContour(hull)[0]

            #self.completeContour(exteriorContour)
            self.saveContour(tree, self.pointsToContour(exteriorContour))

    def refreshTree(self, tree):
        global finalComps

        tree.delete(*tree.get_children()) # kill all the children

        for i, comp in enumerate(finalComps):
            head = tree.insert("", i, text=str(comp))
            for x, subComp in enumerate(finalComps[comp]):
                tree.insert(head, x, text=(str(comp) + " " + str(x)), values="Saved")

    def refreshTreeOld(self, tree):
        global componentList, stentComponents

        tree.delete(*tree.get_children()) # kill all the children
        for i in range(0, len(componentList)):
            # Custom for stents (since there are 1-10 of them)
            if componentList[i][0] == "Stents":
                tree.insert("", i, "Stents", text=componentList[i][0])
                for i in range(0, len(stentComponents)):
                    tree.insert("Stents", i, text=stentComponents[i][0], values=stentComponents[i][1])
            else:
                tree.insert("", i, text=componentList[i][0], values=componentList[i][1])

    def clearPoints(self, canvas):
        global points
        
        points.clear()
        canvas.delete("point")

    def paint(self, x, y, canvas):
        global pointCount
        
        #print("painting", points)
        x1, y1 = (x - 3), (y - 3)
        x2, y2 = (x + 3), (y + 3)
        canvas.create_oval(x1, y1, x2, y2, fill = "#ff0000", tags=("point", pointCount))

    def refreshCanvas(self, canvas):
        global points

        canvas.delete("point")
        for point in points:
            self.paint(point[0], point[1], canvas)

    def addComponent(self):
        global finalComps
        inputDialog = tkinter.simpledialog.askstring("Add new component", "Component Name:")
        finalComps.update({inputDialog : []})

    def removeComponent(self, tree):
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

    def getDist(self, pointA, pointB):
        x1 = pointA[0]
        y1 = pointA[1]
        x2 = pointB[0]
        y2 = pointB[1]

        return(sqrt((x2 - x1)**2 + (y2 - y1)**2))

    def orderPoints(self, points):
        global componentList

        newPoints = []
        oldPoints = points.copy()

        if len(oldPoints) > 0:
            newPoints.append(oldPoints.pop())

        while len(oldPoints) > 0:
            currentPoint = newPoints[-1]
            nextPoint = None
            smallestDist = 999999999999 # lmao praying the distance between two points isn't over a trillion
            for checkPoint in oldPoints:
                dist = self.getDist(currentPoint, checkPoint)
                if dist < smallestDist:
                    nextPoint = checkPoint
                    smallestDist = dist

            newPoints.append(nextPoint)
            oldPoints.remove(nextPoint)

        return newPoints

    def completeContour(self, points):
        global pathToImg

        img = cv2.imread(pathToImg)

        cv2.drawContours(img, self.orderPoints(self.pointsToContour(points)), 0, (0,0,255), 2)

        cv2.imshow("Manual Contour", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def pointsToContour(self, points):
        a = np.array(points)
        return([a])

    def saveContour(self, tree, contour):
        global componentList, points, finalComps, currentName
        
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

        self.refreshTree(tree)
    
    def viewContour(self, tree):
        global pathToImg, finalComps

        selected = tree.focus()
        currentComponent = tree.item(selected)
        componentName = currentComponent["text"]

        compParent = componentName.split(" ")[0]
        if len(componentName.split(" ")) > 1:
            compId = componentName.split(" ")[1]

        if compParent in finalComps:
            if len(componentName.split(" ")) == 1:
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

    def textProcess(self):
        global componentList

        outputFile = open("output.txt", "w+")

        for i in componentList:
            stentArea = 0
            if str(i[0]) == "Stents":
                for x in stentComponents:
                    if x[1] == "Saved":
                        currentContour = []
                        for y in x[2][0].tolist():
                            if any(isinstance(inst, list) for inst in y):
                                currentContour.append((y[0][0],y[0][1]))
                            else:
                                currentContour.append((y[0],y[1]))

                        area = round(cv2.contourArea(self.pointsToContour(currentContour)[0]), 3)
                        stentArea += area

                outputFile.write("Component Name: " + "Stents" + " | Area: " + str(stentArea) + "\n")

            if i[1] == "Saved":
                currentContour = []
                for x in i[2][0].tolist():
                    if any(isinstance(inst, list) for inst in x):
                        currentContour.append((x[0][0],x[0][1]))
                    else:
                        currentContour.append((x[0],x[1]))

                name = str(i[0])
                area = cv2.contourArea(self.pointsToContour(currentContour)[0])
                length = cv2.arcLength(self.pointsToContour(currentContour)[0], True)

                if conversion != 0:
                    length = length/float(conversion)
                    area = area/(float(conversion)**2)

                outputFile.write("Component Name: " + str(name) + " | Area: " + str(area) + " | Length: " + str(length) + "\n")

        outputFile.close()

    def excelOutput(self, file):
        global finalComps

        workbook = xlsxwriter.Workbook(str(file) + ".xlsx")
        worksheet = workbook.add_worksheet()

        row = 0
        col = 0

        for i in finalComps:
            compArea = 0
            for x in finalComps[i]:
                compArea += cv2.contourArea(self.pointsToContour(x[0])[0])

            if conversion != 0:
                compArea = round(compArea/(float(conversion)**2), 3)

            finalVals.update({i : (compArea)}) #TODO include length

        print(finalVals)

    def excelOutputOld(self, file):
        global componentList, stentComponents, finalComps

        workbook = xlsxwriter.Workbook(str(file) + ".xlsx")
        worksheet = workbook.add_worksheet()

        row = 0
        col = 0

        for i in componentList:
            stentArea = 0
            if str(i[0]) == "Stents":
                for x in stentComponents:
                    if x[1] == "Saved":
                        currentContour = []
                        for y in x[2][0].tolist():
                            if any(isinstance(inst, list) for inst in y):
                                currentContour.append((y[0][0],y[0][1]))
                            else:
                                currentContour.append((y[0],y[1]))

                        area = round(cv2.contourArea(self.pointsToContour(currentContour)[0]), 3)

                        if conversion != 0:
                            area = round(area/(float(conversion)**2), 3)

                        stentArea += area

                worksheet.write(0, col, ("Stents" + " a"))
                worksheet.write(1, col, stentArea)
                finalComps.update({"Stents" : stentArea})

                col += 1

            if i[1] == "Saved":
                currentContour = []
                for x in i[2][0].tolist():
                    if any(isinstance(inst, list) for inst in x):
                        currentContour.append((x[0][0],x[0][1]))
                    else:
                        currentContour.append((x[0],x[1]))

                name = str(i[0])
                area = round(cv2.contourArea(self.pointsToContour(currentContour)[0]), 3)
                length = round(cv2.arcLength(self.pointsToContour(currentContour)[0], True), 3)

                if conversion != 0:
                    length = round(length/float(conversion), 3)
                    area = round(area/(float(conversion)**2), 3)

                worksheet.write(0, col, (name + " a"))
                worksheet.write(1, col, area)
                finalComps.update({name : (area, length)})

                col += 1

                worksheet.write(0, col, (name + " p"))
                worksheet.write(1, col, length)

                col += 1

        #print(finalComps)

        # somewhat hard coded vales
        worksheet.write(0, col, ("Lumen a Ca"))
        lumenAreaCa = ((finalComps.get("Lumen")[1])**2)/12.5664
        worksheet.write(1, col, str(lumenAreaCa))

        col += 1

        worksheet.write(0, col, ("Media a Ca"))
        mediaAreaCa = ((finalComps.get("Media")[1])**2)/12.5664
        worksheet.write(1, col, str(mediaAreaCa))

        col += 1

        worksheet.write(0, col, ("Neointima a Ca"))
        neoAreaCa = ((finalComps.get("Neointima")[1])**2)/12.5664
        worksheet.write(1, col, str(neoAreaCa))

        col += 1

        worksheet.write(0, col, ("% Stenosis"))
        stenosis = ((finalComps.get("Neointima")[0] - finalComps.get("Stents") - finalComps.get("Lumen")[0])/(finalComps.get("Neointima")[0]))*100
        worksheet.write(1, col, str(stenosis))

        col += 1

        worksheet.write(0, col, ("% Stenosis Ca"))
        stenosisAreaCa = ((neoAreaCa - finalComps.get("Stents") - lumenAreaCa)/neoAreaCa)*100
        worksheet.write(1, col, str(stenosisAreaCa))

        col += 1

        worksheet.write(0, col, ("N/M"))
        nm = (finalComps.get("Neointima")[0] - finalComps.get("Stents") - finalComps.get("Lumen")[0])/(finalComps.get("Media")[0] - finalComps.get("Neointima")[0])
        worksheet.write(1, col, str(nm))

        col += 1

        workbook.close()

    def manualContour(self, tree):
        global pathToImg, pointCount, points, finalComps

        points.clear()

        def preErase(event):
            global pointCount
            x = event.x
            y = event.y

            closestPoint = None
            shortestDist = 9999999999999
            for point in points:
                dist = self.getDist((x,y), point)
                if dist < shortestDist:
                    shortestDist = dist
                    closestPoint = point
            
            print(closestPoint)
            points.remove(closestPoint)
            self.refreshCanvas(canvas)

            #points.append((x, y))
            #pointCount += 1
            
            #self.paint(x, y, canvas)

        def prePaint(event):
            global pointCount
            x = event.x 
            y = event.y

            points.append((x, y))
            pointCount += 1
            
            self.paint(x, y, canvas)

        def undo(event):
            global pointCount, points 
            
            if len(points) > 0:
                print("undoing...")

                points.pop()
                pointCount -= 1

                print(points)

                self.refreshCanvas(canvas)
            
            else:
                print("Cannot undo, no points on canvas.")

        img = ImageTk.PhotoImage(Image.open(pathToImg))
        self.pack(fill=BOTH, expand=True)
        manCont = tk.Toplevel()
        manCont.wm_title("Manual Contour")

        canvas = tkinter.Canvas(manCont, width=img.width(), height=img.height())
        canvas.grid(row=0, column=0, rowspan = 5, sticky=N+S+E+W)

        canvas.create_image(0, 0, image=img, anchor="nw")

        # mouseclick events
        canvas.bind("<Button 1>", prePaint)
        canvas.bind("<Button 3>", undo)

        def eraseMode():
            canvas.bind("<Button 1>", preErase)

        def addMode():
            canvas.bind("<Button 1>", prePaint)

        add = tk.Button(manCont, text="Add Mode", command = lambda:[addMode()])
        add.grid(row=0, column=1, padx=10, pady=10, sticky=E+W+S+N)

        erase = tk.Button(manCont, text="Erase Mode", command = lambda:[eraseMode()])
        erase.grid(row=1, column=1, padx=10, pady=10, sticky=E+W+S+N)
        
        clearContour = tk.Button(manCont, text="Clear Points", command = lambda:[self.clearPoints(canvas)])
        clearContour.grid(row=2, column=1, padx=10, pady=10, sticky=E+W+S+N)

        complete = tk.Button(manCont, text="Complete Contour", command = lambda:[self.completeContour(self.orderPoints(points))])
        complete.grid(row=3, column=1, padx=10, pady=10, sticky=E+W+S+N)

        saveContour = tk.Button(manCont, text="Save Contour", command = lambda:[self.saveContour(tree, self.pointsToContour(self.orderPoints(points))), manCont.destroy()])
        saveContour.grid(row=4, column=1, padx=10, pady=10, sticky=E+W+S+N)

        selected = tree.focus()
        currentComponent = tree.item(selected)
        componentName = currentComponent["text"]

        compParent = componentName.split(" ")[0]
        if len(componentName.split(" ")) > 1:
            compId = componentName.split(" ")[1]

        if compParent in finalComps:
            if len(componentName.split(" ")) > 1:
                contour = (finalComps[compParent])[int(compId)]
                for i in contour[0]:
                    points.append((i.tolist()[0][0], i.tolist()[0][1]))
                    self.paint(i.tolist()[0][0], i.tolist()[0][1], canvas)

        manCont.mainloop()
    
    def manualContourOld(self, tree):
        global pathToImg, pointCount, points, componentList

        def preErase(event):
            global pointCount
            x = event.x 
            y = event.y

            closestPoint = None
            shortestDist = 9999999999999
            for point in points:
                dist = self.getDist((x,y), point)
                if dist < shortestDist:
                    shortestDist = dist
                    closestPoint = point
            
            print(closestPoint)
            points.remove(closestPoint)
            self.refreshCanvas(canvas)

            #points.append((x, y))
            #pointCount += 1
            
            #self.paint(x, y, canvas)

        def prePaint(event):
            global pointCount
            x = event.x 
            y = event.y

            points.append((x, y))
            pointCount += 1
            
            self.paint(x, y, canvas)

        def undo(event):
            global pointCount, points 
            
            if len(points) > 0:
                print("undoing...")

                points.pop()
                pointCount -= 1

                print(points)

                self.refreshCanvas(canvas)
            
            else:
                print("Cannot undo, no points on canvas.")

        img = ImageTk.PhotoImage(Image.open(pathToImg))
        self.pack(fill=BOTH, expand=True)
        manCont = tk.Toplevel()
        manCont.wm_title("Manual Contour")

        canvas = tkinter.Canvas(manCont, width=img.width(), height=img.height())
        canvas.grid(row=0, column=0, rowspan = 5, sticky=N+S+E+W)

        canvas.create_image(0, 0, image=img, anchor="nw")

        # mouseclick events
        canvas.bind("<Button 1>", prePaint)
        canvas.bind("<Button 3>", undo)

        def eraseMode():
            canvas.bind("<Button 1>", preErase)

        def addMode():
            canvas.bind("<Button 1>", prePaint)

        add = tk.Button(manCont, text="Add Mode", command = lambda:[addMode()])
        add.grid(row=0, column=1, padx=10, pady=10, sticky=E+W+S+N)

        erase = tk.Button(manCont, text="Erase Mode", command = lambda:[eraseMode()])
        erase.grid(row=1, column=1, padx=10, pady=10, sticky=E+W+S+N)
        
        clearContour = tk.Button(manCont, text="Clear Points", command = lambda:[self.clearPoints(canvas)])
        clearContour.grid(row=2, column=1, padx=10, pady=10, sticky=E+W+S+N)

        complete = tk.Button(manCont, text="Complete Contour", command = lambda:[self.completeContour(self.orderPoints(points))])
        complete.grid(row=3, column=1, padx=10, pady=10, sticky=E+W+S+N)

        saveContour = tk.Button(manCont, text="Save Contour", command = lambda:[self.saveContour(tree, self.pointsToContour(self.orderPoints(points))), manCont.destroy()])
        saveContour.grid(row=4, column=1, padx=10, pady=10, sticky=E+W+S+N)

        selected = tree.focus()
        currentComponent = tree.item(selected)
        componentName = currentComponent["text"]

        if componentName.split()[0] == "Stent":
            for x in stentComponents:
                if x[0] == componentName and x[1] == "Saved":
                    for z in x[2][0].tolist():
                        if any(isinstance(inst, list) for inst in z):
                            points.append((z[0][0],z[0][1]))
                            self.paint(z[0][0], z[0][1], canvas)
                        else:
                            points.append((z[0],z[1]))
                            self.paint(z[0], z[1], canvas)
        else:
            for i in componentList:
                if i[0] == componentName and i[1] == "Saved":
                    for x in i[2][0].tolist():
                        if any(isinstance(inst, list) for inst in x):
                            points.append((x[0][0],x[0][1]))
                            self.paint(x[0][0], x[0][1], canvas)
                        else:
                            points.append((x[0],x[1]))
                            self.paint(x[0], x[1], canvas)

        manCont.mainloop()

    def dataExtraction(self):
        self.pack(fill=BOTH, expand=True)
        popup = tk.Toplevel()
        popup.wm_title("Data Extraction")

        tree=ttk.Treeview(popup)
        tree.grid(row=0, column=0, columnspan=5, rowspan=3, padx=10, pady=10, sticky=E+W+S+N)

        tree["columns"]=("one")
        tree.column("#0", width=270, minwidth=270, stretch=tk.NO)
        tree.column("one", width=150, minwidth=150, stretch=tk.NO)

        tree.heading("#0",text="Component",anchor=tk.W)
        tree.heading("one", text="Status",anchor=tk.W)

        self.refreshTree(tree)

        add = tk.Button(popup, text="Add", command = lambda:[self.addComponent(), self.refreshTree(tree)])
        add.grid(row=3, column=0, padx=10, pady=10, sticky=E+W+S+N)

        delete = tk.Button(popup, text="Remove", command = lambda:[self.removeComponent(tree), self.refreshTree(tree)])
        delete.grid(row=3, column=1, padx=10, pady=10, sticky=E+W+S+N)

        flood = tk.Button(popup, text="Supervised", command = lambda:[self.floodFill(tree)])
        flood.grid(row=3, column=2, padx=10, pady=10, sticky=E+W+S+N)
        
        manual = tk.Button(popup, text="Manual Contour", command = lambda:[self.manualContour(tree)])
        manual.grid(row=3, column=3, padx=10, pady=10, sticky=E+W+S+N)

        view = tk.Button(popup, text="View Contour", command = lambda:[self.viewContour(tree)])
        view.grid(row=3, column=5, padx=10, pady=10, sticky=E+W+S+N)

        exportLabel = tk.Label(popup, text="Export ", font=("Helvetica 12 bold"))
        exportLabel.grid(row=0, column = 5, padx=20, pady=10, sticky=E+W+S+N)

        text = tk.Button(popup, text=".txt", command = lambda:[self.textProcess(), popup.destroy()])
        text.grid(row=1, column=5, padx=10, pady=10, sticky=E+W+S+N)

        excel = tk.Button(popup, text=".xls", command = lambda:[self.outputLoc(), popup.destroy()])
        excel.grid(row=2, column=5, padx=10, pady=10, sticky=E+W+S+N)
        
        popup.mainloop()

    def experimental(self):
        self.pack(fill=BOTH, expand=True)
        popup = tk.Toplevel()
        popup.wm_title("Experimental")

        exportLabel = tk.Label(popup, text="CNN Tuning & Launch", font=("Helvetica 12 bold"))
        exportLabel.grid(row=0, column=0, columnspan=2, padx=20, pady=10, sticky=E+W+S+N)

        btnTune = Button(popup, text="Manual Tuning", command = lambda:[self.tune()])
        btnTune.grid(row=1, column=0, padx=5, pady=10, sticky=E+W+S+N)

        btnSegment = Button(popup, text="Start CNN", command = lambda:[self.segment()])
        btnSegment.grid(row=1, column=1, padx=5, pady=10, sticky=E+W+S+N)

        popup.mainloop()

    def initUI(self):
        self.master.title("Segmentation")
        self.pack(fill=BOTH, expand=True)

        self.columnconfigure(1, weight=1)
        #self.columnconfigure(3, pad=7)
        #self.rowconfigure(1, weight=1)
        #self.rowconfigure(5, pad=7)
        
        panel = tk.Label(self, text = "Image will appear here.", bg="white")
        #panel.image = photoImg
        panel.grid(row=1, column=0, columnspan=2, rowspan=5, padx=10, pady=10, sticky=E+W+S+N)

        btnSelect = Button(self, text="Select an Image", command = self.selectImage)
        btnSelect.grid(row=1, column=3, padx=10, pady=20, sticky=E+W+S+N)

        btnAdjust = Button(self, text="Image Adjustments", command = self.chooseAdjust)
        btnAdjust.grid(row=2, column=3, padx=10, pady=20, sticky=E+W+S+N)

        btnCalibrate = Button(self, text="Calibration", command = self.calibration)
        btnCalibrate.grid(row=3, column=3, padx=10, pady=20, sticky=E+W+S+N)
        
        btnPartition = Button(self, text="Data Extraction", command = lambda:[self.dataExtraction()])
        btnPartition.grid(row=4, column=3, padx=10, pady=20, sticky=E+W+S+N)

        btnExp = Button(self, text="Experimental Mode", command = lambda:[self.experimental()])
        btnExp.grid(row=5, column=3, padx=10, pady=20, sticky=E+W+S+N)
        
def main():
    root = Tk()
    root.geometry("350x325")
    #root.geometry("")
    app = Segmentation()
    root.mainloop()

if __name__ == '__main__':
    main()