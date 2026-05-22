import random
import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

MATRIX_SIZE = 26

COLORMAP_COOLTOWARM = mpl.colors.ListedColormap(mpl.colormaps['coolwarm'](np.linspace(0.0, 1.0, 256)))
COLORMAP_GREENTORED = mpl.colors.ListedColormap(mpl.colormaps['RdYlGn'](np.linspace(1.0, 0.0, 256)))

# print(COLORMAP_COOLTOWARM.colors[128])

class TempMatrixViewer:
    def __init__(self, root):

        # WBT_PSY = np.load("WBGT project\\WBT_PSYdb.npy")
        # WBT_ISO = np.load("WBGT project\\WBT_ISOdb.npy")
        # WBT_LIL = np.load("WBGT project\\WBT_LILdb.npy")
        # WBGT_PSY = np.load("WBGT project\\WBGT_PSYdb.npy")
        # WBGT_ISO = np.load("WBGT project\\WBGT_ISOdb.npy")
        WBGT_LIL = np.load("WBGT_LILdb.npy")
        # print(WBGT_LIL)

        bodyframe = tk.Frame(root)
        bodyframe.grid(column=0, row=0, sticky=(tk.W,tk.E))

        axisSelectFrame = tk.Frame(bodyframe)
        axisSelectFrame.grid(row=0, column=0, sticky="EW", padx=0, pady=0)
        axisSelectLabel = tk.Label(axisSelectFrame, text="Choose data to display:")
        axisSelectLabel.grid(row=0, column=0, columnspan=3, padx=0, pady=0)

        # ((taMax, rhMax, bpMax, tgMax, vaMax, rsMax))
        xSelectFrame = tk.Frame(axisSelectFrame)
        xSelectFrame.grid(row=1, column=0, sticky="EW", padx=0, pady=0)
        xSelectLabel = tk.Label(xSelectFrame, text="X axis:")
        xSelectLabel.grid(row=0, column=0, sticky="EW", padx=0, pady=0)
        xAxis = tk.IntVar().set(-1)
        xDB = tk.IntVar().set(0)
        xRH = tk.IntVar().set(0)
        xBP = tk.IntVar().set(0)
        xGT = tk.IntVar().set(0)
        xAV = tk.IntVar().set(0)
        xSR = tk.IntVar().set(0)
        buttonXDB = tk.Checkbutton(xSelectFrame, text="DB", variable=xDB)
        buttonXDB.grid(row=1, column=0, sticky="W", padx=0, pady=0)
        buttonXRH = tk.Checkbutton(xSelectFrame, text="RH", variable=xRH)
        buttonXRH.grid(row=2, column=0, sticky="W", padx=0, pady=0)
        buttonXBP = tk.Checkbutton(xSelectFrame, text="BP", variable=xBP)
        buttonXBP.grid(row=3, column=0, sticky="W", padx=0, pady=0)
        buttonXGT = tk.Checkbutton(xSelectFrame, text="GT", variable=xGT)
        buttonXGT.grid(row=4, column=0, sticky="W", padx=0, pady=0)
        buttonXAV = tk.Checkbutton(xSelectFrame, text="AV", variable=xAV)
        buttonXAV.grid(row=5, column=0, sticky="W", padx=0, pady=0)
        buttonXSR = tk.Checkbutton(xSelectFrame, text="SR", variable=xSR)
        buttonXSR.grid(row=6, column=0, sticky="W", padx=0, pady=0)

        ySelectFrame = tk.Frame(axisSelectFrame)
        ySelectFrame.grid(row=1, column=1, sticky="EW", padx=0, pady=0)
        ySelectLabel = tk.Label(ySelectFrame, text="Y axis:")
        ySelectLabel.grid(row=0, column=0, sticky="EW", padx=0, pady=0)
        yAxis = tk.IntVar().set(-1)
        yDB = tk.IntVar().set(0)
        yRH = tk.IntVar().set(0)
        yBP = tk.IntVar().set(0)
        yGT = tk.IntVar().set(0)
        yAV = tk.IntVar().set(0)
        ySR = tk.IntVar().set(0)
        buttonYDB = tk.Checkbutton(ySelectFrame, text="DB", variable=xDB)
        buttonYDB.grid(row=2, column=0, sticky="W", padx=0, pady=0)
        buttonYRH = tk.Checkbutton(ySelectFrame, text="RH", variable=xRH)
        buttonYRH.grid(row=3, column=0, sticky="W", padx=0, pady=0)
        buttonYBP = tk.Checkbutton(ySelectFrame, text="BP", variable=xBP)
        buttonYBP.grid(row=4, column=0, sticky="W", padx=0, pady=0)
        buttonYGT = tk.Checkbutton(ySelectFrame, text="GT", variable=xGT)
        buttonYGT.grid(row=5, column=0, sticky="W", padx=0, pady=0)
        buttonYAV = tk.Checkbutton(ySelectFrame, text="AV", variable=xAV)
        buttonYAV.grid(row=6, column=0, sticky="W", padx=0, pady=0)
        buttonYSR = tk.Checkbutton(ySelectFrame, text="SR", variable=xSR)
        buttonYSR.grid(row=7, column=0, sticky="W", padx=0, pady=0)

        zSelectFrame = tk.Frame(axisSelectFrame)
        zSelectFrame.grid(row=1, column=2, sticky="EW", padx=0, pady=0)
        zSelectLabel = tk.Label(zSelectFrame, text="Z axis:")
        zSelectLabel.grid(row=0, column=0, sticky="EW", padx=0, pady=0)
        zAxis = tk.IntVar().set(-1)
        zDB = tk.IntVar().set(0)
        zRH = tk.IntVar().set(0)
        zBP = tk.IntVar().set(0)
        zGT = tk.IntVar().set(0)
        zAV = tk.IntVar().set(0)
        zSR = tk.IntVar().set(0)
        buttonZDB = tk.Checkbutton(zSelectFrame, text="DB", variable=xDB)
        buttonZDB.grid(row=2, column=0, sticky="W")
        buttonZRH = tk.Checkbutton(zSelectFrame, text="RH", variable=xRH)
        buttonZRH.grid(row=3, column=0, sticky="W")
        buttonZBP = tk.Checkbutton(zSelectFrame, text="BP", variable=xBP)
        buttonZBP.grid(row=4, column=0, sticky="W")
        buttonZGT = tk.Checkbutton(zSelectFrame, text="GT", variable=xGT)
        buttonZGT.grid(row=5, column=0, sticky="W")
        buttonZAV = tk.Checkbutton(zSelectFrame, text="AV", variable=xAV)
        buttonZAV.grid(row=6, column=0, sticky="W")
        buttonZSR = tk.Checkbutton(zSelectFrame, text="SR", variable=xSR)
        buttonZSR.grid(row=7, column=0, sticky="W")

        # selectXFrame = tk.Label(dataSelectFrame, text="X axis:")
        # selectXFrame.grid(row=1, column=1)

        dataFrame = tk.Frame(bodyframe)
        dataDisplay = tk.Canvas(bodyframe)
        dataDisplay.grid(row=0, column=1, sticky="EW")
        s = round(800.0/26.0)
        for i in range(0,MATRIX_SIZE+1):
            for j in range(0,MATRIX_SIZE+1):
                # dataDisplay.create_rectangle(i*4,j*4,4,4, fill=("#" + str(i) + "00" + str(j)))
                dataDisplay.create_rectangle(i*10,j*10,i*10+10,j*10+10, width=0, fill=("#" + str(random.randint(10,99)) + str(random.randint(10,99)) + str(random.randint(10,99))))

        x = tk.IntVar()
        y = tk.IntVar()
        z = tk.IntVar()
        zSlider = tk.Scale(bodyframe, from_=100, to_=0)
        zSlider.grid(row=0, column=2, sticky="E")

        bulbSelectFrame = tk.LabelFrame(bodyframe)
        bulbSelectFrame.grid(row=2, column=1)
        tk.Label(bulbSelectFrame, text="Select Data:", justify=tk.LEFT)
        m = tk.IntVar().set(1)
        # m.set(1) # initialize
        mSelect = tk.IntVar()
        matrices = [("Tnwb"), ("WBGT")]
        for matrix in matrices:
            tk.ttk.Radiobutton(bulbSelectFrame, text=matrix, variable=m, command=self.drawMatrix, value=mSelect)


    
    def loadMatrix(self, *args):
        pass
    def drawMatrix(self, xParam, yParam, zParam, *args):
        pass

root = tk.Tk()
root.title("Bulb Temp Matrix Viewer")
root.geometry("900x900+80+40")
TempMatrixViewer(root)
root.mainloop()