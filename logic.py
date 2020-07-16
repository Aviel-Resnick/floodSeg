'''
Aviel Resnick, 2020
Utility designed for the automated, supervised, or manual morphometry of images, particularly stent-implanted coronary arteries.

main.py - Computations, misc. functions, logic
'''
import tkinter as tk
import tkinter.filedialog
import tkinter.simpledialog

#finalComps = {"Lumen" : [], "Neointima" : [], "Media" : [], "Stents" : [], "Thickness" : []}
finalComps = {}

def addComponent(tree):
    global finalComps

    inputDialog = tk.simpledialog.askstring("Add new component", "Component Name:")
    finalComps.update({inputDialog : []})
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

    