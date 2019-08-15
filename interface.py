import tkinter as tk
import tkinter.filedialog
import tkinter.simpledialog
from PIL import ImageTk, Image
from tkinter import Tk, Text, BOTH, W, N, E, S
from tkinter.ttk import Frame, Button, Label, Style
import os.path
from os import path
import webbrowser
import unsupervisedSeg
pathToImg = None
configFile = None

# Known bugs:
#   Duplicate Colors
#   Not parsing args from file
#   Window scales up but not down

# To Do:
#   Implement image adjustment
#   Comment sections
#   Design and implement component identification
#   Area to Metric conversion
#   Output data to xls
#   Improve parameters

class Segmentation(Frame):
    def __init__(self):
        super().__init__()
        self.initUI()

    def selectImage(self):
        global pathToImg
        pathToImg = tkinter.filedialog.askopenfilename(filetypes=[("Image File",'.jpg')])
        img = Image.open(pathToImg)
        photoImg = ImageTk.PhotoImage(img)
        width, height = img.size
        resX = width + 150 if width + 150 > 350 else 350
        resY = height + 20 if height + 20 > 390 else 390
        
        #self.master.geometry(str(resX)+"x"+str(resY))
        
        panel = tk.Label(self, image = photoImg)
        panel.image = photoImg
        panel.grid(row=1, column=0, columnspan=2, rowspan=6, padx=5, sticky=E+W+S+N)
        
        self.master.geometry("")
        
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
        inputFile = pathToImg
        print(inputFile)
        unsupervisedSeg.main(inputFile, configFile)

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
        btnSelect.grid(row=1, column=3, padx=10, pady=20)

        btnAdjust = Button(self, text="Image Adjustments", command = self.selectImage)
        btnAdjust.grid(row=2, column=3, padx=10, pady=20)

        btnTune = Button(self, text="Manual Tuning", command = self.tune)
        btnTune.grid(row=3, column=3, padx=10, pady=20)

        btnCalibrate = Button(self, text="Calibration", command = self.calibration)
        btnCalibrate.grid(row=4, column=3, padx=10, pady=20)

        btnSegment = Button(self, text="Train & Segment", command = self.segment)
        btnSegment.grid(row=5, column=3, padx=10, pady=20)

        btnPartition = Button(self, text="Assign Partitions")
        btnPartition.grid(row=6, column=3, padx=10, pady=20)

def main():
    root = Tk()
    #root.geometry("350x390")
    root.geometry("")
    app = Segmentation()
    root.mainloop()

if __name__ == '__main__':
    main()