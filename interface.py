'''
Aviel Resnick, 2019
Utility designed for the automated segmentation of images, particulary stented coronary arteries.
'''

import tkinter as tk
import tkinter.filedialog
import tkinter.simpledialog
from PIL import ImageTk, Image, ImageEnhance
from tkinter import Tk, Text, BOTH, W, N, E, S
from tkinter.ttk import Frame, Button, Label, Style
import os.path
from os import path
import webbrowser
import unsupervisedSeg

img = None # original
pathToImg = None
currentImage = None
newImagePath = None
configFile = None

'''
Known bugs:
    Duplicate Colors
    Not parsing args from file
    Window scales up but not down

To Do:
    Comment sections
    Design and implement component identification
    Area to Metric conversion
    Output data to xls
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
            print("dsfs")
            newImagePath = pathToImg
        #inputFile = pathToImg
        #print(inputFile)
        unsupervisedSeg.main(newImagePath, pathToImg, configFile)

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

        btnPartition = Button(self, text="Assign Partitions")
        btnPartition.grid(row=6, column=3, padx=10, pady=20, sticky=E+W+S+N)

def main():
    root = Tk()
    root.geometry("350x390")
    #root.geometry("")
    app = Segmentation()
    root.mainloop()

if __name__ == '__main__':
    main()