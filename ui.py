'''
Aviel Resnick, 2020
Utility designed for the automated, supervised, or manual morphometry of images, particularly stent-implanted coronary arteries.

ui.py - main GUI

Color Palette:
    Light Grey #535353
    Medium Grey #464646
    Dark Grey #262626
    Off-White #F5F5F5
'''

import tkinter as tk
import tkinter.filedialog
import tkinter.simpledialog
from tkinter import ttk
from tkinter import Tk, Text, BOTH, W, N, E, S
from tkinter.ttk import Frame, Button, Label, Style
from PIL import ImageTk, Image, ImageEnhance, ImageDraw
import logic

class MainGUI(Frame):
    conversion = 0
    frameReference=0
    framePreview=0
    tree=0

    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.master.title("floodSeg")

        # REFERENCE IMAGE FRAME

        frameReference=tk.LabelFrame(self.master, width=700, height=550, bg="#F5F5F5", bd=4, text="Reference Image")
        frameReference.grid(row=0, column=0, padx=(100, 100), pady=(50,50))

        panelRef = tk.Label(frameReference, text = "Image will appear here.", bg="#F5F5F5", padx=280, pady=260)
        panelRef.pack()

        # PREVIEW IMAGE FRAME

        framePreview=tk.LabelFrame(self.master, width=700, height=550, bg="#F5F5F5", bd=4,  text="Dynamic Preview")
        framePreview.grid(row=0, column=1, padx=(100, 100), pady=(50,50))

        panelPreview = tk.Label(framePreview, text = "Image will appear here.", bg="#F5F5F5", padx=280, pady=260)
        panelPreview.pack()

        # COMPONENT TABLE FRAME

        frameComponentTable=tk.LabelFrame(self.master, width=960, height=380, bg="#F5F5F5", bd=4, text="Component Table")
        frameComponentTable.grid(row=1, column=0, padx=(0, 0), pady=(0,0))

        MainGUI.tree=ttk.Treeview(frameComponentTable)
        frameComponentTable.grid_propagate(False)
        MainGUI.tree.grid(row=0, column=0, columnspan=1, rowspan=5, padx=10, pady=10)

        MainGUI.tree["columns"]=("#1","#2")
        MainGUI.tree.column("#0", width=350, minwidth=350, stretch=tk.NO)
        MainGUI.tree.column("#1", width=200, minwidth=200, stretch=tk.NO)
        MainGUI.tree.column("#2", width=200, minwidth=200, stretch=tk.NO)

        MainGUI.tree.heading("#0",text="Component",anchor=tk.W)
        MainGUI.tree.heading("#1",text="Status",anchor=tk.W)
        MainGUI.tree.heading("#2",text="Preview", anchor=tk.W)

        add = tk.Button(frameComponentTable, text="Add Component", width = 20, command = lambda:[logic.addComponent(MainGUI.tree)])
        add.grid(row=0, column=1, padx=10, pady=10)

        delete = tk.Button(frameComponentTable, text="Remove Component", width = 20, command = lambda:[print("Removing Component")])
        delete.grid(row=1, column=1, padx=10, pady=10)

        flood = tk.Button(frameComponentTable, text="Assisted Contour", width = 20, command = lambda:[print("Assisted Contour")])
        flood.grid(row=2, column=1, padx=10, pady=10)
        
        manual = tk.Button(frameComponentTable, text="Manual Contour", width = 20, command = lambda:[print("Manual Contour")])
        manual.grid(row=3, column=1, padx=10, pady=10)

        view = tk.Button(frameComponentTable, text="Toggle Preview", width = 20, command = lambda:[print("Toggle Preview")])
        view.grid(row=4, column=1, padx=10, pady=10)

        # DATA OUTPUT FRAME

        frameOutput=tk.LabelFrame(self.master, width=960, height=380, bg="#F5F5F5", bd=4, text="Data Output")
        frameOutput.grid(row=1, column=1, padx=(0, 0), pady=(0,0))

    def calibration():
        MainGUI.conversion = tkinter.simpledialog.askstring("Calibration", "1mm = ?px")

    def presets():
        print(MainGUI.tree)
        logic.presets(MainGUI.tree, "Arterial Segmentation")

    def selectImage():
        pathToImg = tkinter.filedialog.askopenfilename(filetypes=[("Image File",'.jpg')])
        img = Image.open(pathToImg)
        photoImg = ImageTk.PhotoImage(img)
        
        MainGUI.updateReferenceImage(photoImg)
        MainGUI.updatePreviewImage(photoImg)
        print(MainGUI.conversion)

    def updateReferenceImage(image):
        panelRef = tk.Label(MainGUI.frameReference, image = image)
        panelRef.image = image
        panelRef.grid(row=0, column=0)

    def updatePreviewImage(image):
        panelPreview = tk.Label(MainGUI.framePreview, image = image)
        panelPreview.image = image
        panelPreview.grid(row=0, column=1)

def main():
    root = Tk()
    root.state("zoomed")
    root.configure(bg='#F5F5F5')
    app = MainGUI()
    
    # MENUBAR
    menubar = tk.Menu(root)
    menubar.add_command(label="Open", command=MainGUI.selectImage)
    menubar.add_separator()

    editmenu = tk.Menu(menubar, tearoff=0)
    editmenu.add_command(label="Calibration", command=MainGUI.calibration)
    editmenu.add_separator()
    editmenu.add_command(label="Image Adjustments", command=root.quit)
    editmenu.add_separator()

    presetmenu = tk.Menu(editmenu, tearoff=0)
    presetmenu.add_command(label="Arterial Segmentation", command=MainGUI.presets)
    editmenu.add_cascade(label="Presets", menu=presetmenu)

    menubar.add_cascade(label="Edit", menu=editmenu)
    menubar.add_separator()
    menubar.add_command(label="Export", command=root.quit)
    menubar.add_separator()
    menubar.add_command(label="About", command=root.quit)
    menubar.add_separator()
    menubar.add_command(label="Help", command=root.quit)
    menubar.add_separator()
    menubar.add_command(label="Quit", command=root.quit)
    menubar.add_separator()

    # display the menu
    root.config(menu=menubar)

    root.mainloop()


if __name__ == '__main__':
    main()