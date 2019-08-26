'''
Aviel Resnick, 2019
Utility designed for the automated segmentation of images, particulary stented coronary arteries.
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
import unsupervisedSeg
import cv2
import numpy as np
from math import sqrt

img = None # original
pathToImg = None
currentImage = None
newImagePath = None
configFile = None
points = []
pointCount = 0
componentList = [["EEL", "Empty", None],["IEL", "Empty", None], ["Neointima", "Empty", None], ["Lumen", "Empty", None]]

'''
Known bugs:
    Could crash if a file isn't selected, or no new component name is given
    Duplicate Colors
    Not parsing args from file
    Since I am not removing the "deleted" item from componentList, its returns at the next refresh

To Do:
    Finish data extraction section
    Minor adjustments & full manual
    Comment sections
    Design and implement component identification
    Area to Metric conversion
    Improve parameters
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
        global pathToImg, img
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

        self.updateImage(photoImg)
        
        self.master.geometry("")

    def updateImage(self, image):
        global currentImage

        panel = tk.Label(self, image = image)
        panel.image = image
        currentImage = image
        panel.grid(row=1, column=0, columnspan=2, rowspan=6, padx=5, sticky=E+W+S+N)
    
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

        restore = tk.Button(popup, text="Restore to Original", command = lambda:[self.adjust(-1, -1, -1), popup.destroy()])
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
        inputDialog = tkinter.simpledialog.askstring("Calibration", "1mm = ?px")
        print(inputDialog)
     
    def segment(self):
        global pathToImg, newImagePath

        if newImagePath is None:
            newImagePath = pathToImg
        #inputFile = pathToImg
        #print(inputFile)
        unsupervisedSeg.main(newImagePath, pathToImg, configFile)

    def refreshTree(self, tree):
        global componentList

        tree.delete(*tree.get_children()) # kill all the children
        for i in range(0, len(componentList)):
            tree.insert("", i, text=componentList[i][0], values=componentList[i][1])

    def clearPoints(self, canvas):
        global points
        
        points.clear()
        canvas.delete("point")

    def paint(self, x, y, canvas):
        global pointCount
        
        print("painting", points)
        x1, y1 = (x - 3), (y - 3)
        x2, y2 = (x + 3), (y + 3)
        canvas.create_oval(x1, y1, x2, y2, fill = "#ff0000", tags=("point", pointCount))

    def refreshCanvas(self, canvas):
        global points

        canvas.delete("point")
        for point in points:
            self.paint(point[0], point[1], canvas)

    def addComponent(self):
        global componentList
        inputDialog = tkinter.simpledialog.askstring("Add new component", "Component Name:")
        component = [inputDialog, "Empty", None]
        componentList.append(component)

    def removeComponent(self, tree):
        global componentList
        component = tree.selection()[0]
        print(component)
        tree.delete(component)

        #componentList.pop(int(component[-1])-1)
        #self.refreshTree(tree)

    def getDist(self, pointA, pointB):
        x1 = pointA[0]
        y1 = pointA[1]
        x2 = pointB[0]
        y2 = pointB[1]

        return(sqrt((x2 - x1)**2 + (y2 - y1)**2))

    def orderPoints(self, points):
        #global points

        newPoints = []
        oldPoints = points.copy()

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
                    print(smallestDist)
            newPoints.append(nextPoint)
            oldPoints.remove(nextPoint)

        print("new points", newPoints)
        #points = newPoints
        return newPoints

    def completeContour(self, points, manCont):
        global pathToImg

        img = cv2.imread(pathToImg)
        cv2.drawContours(img, self.pointsToContour(points), 0, (0,0,255), 2)

        cv2.imshow("Manual Contour", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def pointsToContour(self, points):
        a = np.array(points)
        return([a])

    def saveContour(self, tree, contour):
        global componentList, points

        selected = tree.focus()
        currentComponent = tree.item(selected)
        
        componentName = currentComponent["text"]

        for i in componentList:
            if i[0] == componentName:
                i[1] = "Saved"
                i[2] = contour
                points.clear()

        self.refreshTree(tree)

    
    def manualContour(self, tree):
        global pathToImg, pointCount, points

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
        canvas.grid(row=0, column=0, rowspan = 3, sticky=N+S+E+W)

        canvas.create_image(0, 0, image=img, anchor="nw")

        # mouseclick events
        canvas.bind("<Button 1>", prePaint)
        canvas.bind("<Button 3>", undo)

        clearContour = tk.Button(manCont, text="Clear Points", command = lambda:[self.clearPoints(canvas)])
        clearContour.grid(row=0, column=1, padx=10, pady=10, sticky=E+W+S+N)

        complete = tk.Button(manCont, text="Complete Contour", command = lambda:[self.completeContour(self.orderPoints(points), manCont)])
        complete.grid(row=1, column=1, padx=10, pady=10, sticky=E+W+S+N)

        saveContour = tk.Button(manCont, text="Save Contour", command = lambda:[self.orderPoints(canvas), self.saveContour(tree, self.pointsToContour(points)), manCont.destroy()])
        saveContour.grid(row=2, column=1, padx=10, pady=10, sticky=E+W+S+N)

        manCont.mainloop()

    def dataExtraction(self):
        self.pack(fill=BOTH, expand=True)
        popup = tk.Toplevel()
        popup.wm_title("Data Extraction")

        tree=ttk.Treeview(popup)
        tree.grid(row=0, column=0, columnspan=3, rowspan=3, padx=10, pady=10, sticky=E+W+S+N)

        tree["columns"]=("one")
        tree.column("#0", width=270, minwidth=270, stretch=tk.NO)
        tree.column("one", width=150, minwidth=150, stretch=tk.NO)

        tree.heading("#0",text="Component",anchor=tk.W)
        tree.heading("one", text="Status",anchor=tk.W)

        self.refreshTree(tree)

        add = tk.Button(popup, text="Add", command = lambda:[self.addComponent(), self.refreshTree(tree)])
        add.grid(row=3, column=0, padx=10, pady=10, sticky=E+W+S+N)

        delete = tk.Button(popup, text="Remove", command = lambda:[self.removeComponent(tree)])
        delete.grid(row=3, column=1, padx=10, pady=10, sticky=E+W+S+N)
        
        manual = tk.Button(popup, text="Manual Contour", command = lambda:[self.manualContour(tree)])
        manual.grid(row=3, column=2, padx=10, pady=10, sticky=E+W+S+N)

        exportLabel = tk.Label(popup, text="Export ", font=("Helvetica 12 bold"))
        exportLabel.grid(row=0, column = 3, padx=20, pady=10, sticky=E+W+S+N)

        text = tk.Button(popup, text=".txt", command = lambda:[popup.destroy()])
        text.grid(row=1, column=3, padx=10, pady=10, sticky=E+W+S+N)

        excel = tk.Button(popup, text=".xls", command = lambda:[popup.destroy()])
        excel.grid(row=2, column=3, padx=10, pady=10, sticky=E+W+S+N)
        
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
        panel.grid(row=1, column=0, columnspan=2, rowspan=6, padx=10, pady=10, sticky=E+W+S+N)

        btnSelect = Button(self, text="Select an Image", command = self.selectImage)
        btnSelect.grid(row=1, column=3, padx=10, pady=20, sticky=E+W+S+N)

        btnAdjust = Button(self, text="Image Adjustments", command = self.chooseAdjust)
        btnAdjust.grid(row=2, column=3, padx=10, pady=20, sticky=E+W+S+N)

        btnTune = Button(self, text="Manual Tuning", command = self.tune)
        btnTune.grid(row=3, column=3, padx=10, pady=20, sticky=E+W+S+N)

        btnCalibrate = Button(self, text="Calibration", command = self.calibration)
        btnCalibrate.grid(row=4, column=3, padx=10, pady=20, sticky=E+W+S+N)

        btnSegment = Button(self, text="Train & Segment", command = self.segment)
        btnSegment.grid(row=5, column=3, padx=10, pady=20, sticky=E+W+S+N)

        btnPartition = Button(self, text="Data Extraction", command = self.dataExtraction)
        btnPartition.grid(row=6, column=3, padx=10, pady=20, sticky=E+W+S+N)

def main():
    root = Tk()
    root.geometry("350x390")
    #root.geometry("")
    app = Segmentation()
    root.mainloop()

if __name__ == '__main__':
    main()