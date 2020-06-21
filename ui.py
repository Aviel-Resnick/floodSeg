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

class MainGUI(Frame):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.master.title("floodSeg")

        frameReference=tk.LabelFrame(self.master, width=760, height=500, bg="#F5F5F5", bd=4, text="Reference Image")
        frameReference.grid(row=0, column=0, padx=(100, 100), pady=(100,50))

        panelRef = tk.Label(frameReference, text = "Image will appear here.", bg="#F5F5F5", padx=310, pady=230)
        panelRef.pack()

        framePreview=tk.LabelFrame(self.master, width=760, height=500, bg="#F5F5F5", bd=4,  text="Dynamic Preview")
        framePreview.grid(row=0, column=1, padx=(100, 100), pady=(100,50))

        panelPreview = tk.Label(framePreview, text = "Image will appear here.", bg="#F5F5F5", padx=310, pady=230)
        panelPreview.pack()

        frameComponentTable=tk.LabelFrame(self.master, width=960, height=380, bg="#F5F5F5", bd=4, text="Component Table")
        frameComponentTable.grid(row=1, column=0, padx=(0, 0), pady=(50,0))

        frameOutput=tk.LabelFrame(self.master, width=960, height=380, bg="#F5F5F5", bd=4, text="Data Output")
        frameOutput.grid(row=1, column=1, padx=(0, 0), pady=(50,0))

    def selectImage():
        print("Select Image")

        

def main():
    root = Tk()
    #root.geometry("350x325")
    root.state("zoomed")
    root.configure(bg='#F5F5F5')
    app = MainGUI()

    menubar = tk.Menu(root)
    menubar.add_command(label="Open", command=MainGUI.selectImage)
    menubar.add_separator()
    menubar.add_command(label="Calibrate", command=root.quit)
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