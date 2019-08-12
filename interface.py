import tkinter as tk
import tkinter.filedialog
from PIL import ImageTk, Image
from tkinter import Tk, Text, BOTH, W, N, E, S
from tkinter.ttk import Frame, Button, Label, Style
import os.path
from os import path
import webbrowser

class Segmentation(Frame):
    def __init__(self):
        super().__init__()
        self.initUI()

    def selectImage(self):
        path = tkinter.filedialog.askopenfilename(filetypes=[("Image File",'.jpg')])
        img = Image.open(path)
        photoImg = ImageTk.PhotoImage(img)
        width, height = img.size
        resX = width + 150
        resY = height + 10
        self.master.geometry(str(resX)+"x"+str(resY))

        panel = tk.Label(self, image = photoImg)
        panel.image = photoImg
        panel.grid(row=1, column=0, columnspan=2, rowspan=5, padx=5, sticky=E+W+S+N)

    def tune(self):
        if path.exists("config.txt"):
            webbrowser.open("config.txt")

        else:
            print("we don't have a config file")

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
        btnSelect.grid(row=1, column=3, padx=10, pady=20)

        btnTune = Button(self, text="Manual Tuning", command = self.tune)
        btnTune.grid(row=2, column=3, padx=10, pady=20)

        btnCalibrate = Button(self, text="Calibration")
        btnCalibrate.grid(row=3, column=3, padx=10, pady=20)

        btnSegment = Button(self, text="Train & Segment")
        btnSegment.grid(row=4, column=3, padx=10, pady=20)

        btnPartition = Button(self, text="Assign Partitions")
        btnPartition.grid(row=5, column=3, padx=10, pady=20)

def main():
    root = Tk()
    root.geometry("350x320")
    app = Segmentation()
    root.mainloop()

if __name__ == '__main__':
    main()