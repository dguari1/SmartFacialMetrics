# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 23:01:29 2017

@author: Diego L. Guarin -- diego_guarin at meei.harvard.edu
"""

from appJar import gui
from ImageProcessMEEInew import take_from_gui

def press(button):
    if button == "Close":
        app.stop()
    else:
        path = app.getEntry("Path:")
        pos = app.getEntry("Initial Position:")        
        take_from_gui(str(path),int(pos))
        app.stop()


app = gui()
app.addLabel("title", "Welcome to Mass. Eye and Ear - Smart Facial Metrics Tool")
app.setLabelBg("title","lightGray")

app.addLabelEntry("Path:")
app.addLabelEntry("Initial Position:")

app.addButtons(["Start", "Close"], press)

app.go()