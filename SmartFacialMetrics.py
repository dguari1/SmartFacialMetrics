# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 09:12:11 2017

@author: Diego L. Guarin -- diego_guarin at meei.harvard.edu
"""

import tkinter as Tk
from tkinter import filedialog
from tkinter import messagebox
from ImageProcessMEEInew import take_from_gui
import os
import sys


def close_window():
    root.destroy()

def browse_file():
    #fname = filedialog.askopenfilename(filetypes = (("Template files", "*.type"), ("All files", "*")))
    TextPath.delete(0, "end")
    fname = filedialog.askdirectory()
    TextPath.insert(0, fname)
    
def clear_box(event):
    TextInitPos.delete(0,'end')
def start_process():
    
    
    
    if not TextPath.get():
        messagebox.showinfo("Error loading files", "No image folder provided")
        check_1=0 #do not continue 
    else:
        path=TextPath.get()
        #now that we have a path it should be validated
        files = os.listdir(path)
        ext=('.png', '.jpg', '.jpeg', '.bmp', '.PNG', '.JPG', '.JPEG', '.BMP')
        files_to_read = [i for i in files if i.endswith(tuple(ext))]
        if len(files_to_read) == 0:
            messagebox.showinfo("Error loading files", "No valid images in folder. Valid formats include .png, .jpg, .jpge and .bmp")
        
        check_1=1 #continue 
    
    
    if not TextInitPos.get():
        messagebox.showinfo("Error loading files", "No initial position provided")
        check_2=0 #do not continue 
    else:
        pos=TextInitPos.get()
        check_2=1 #continue
        
    if check_1 is not 0 and check_2 is not 0:
        root.destroy()
        take_from_gui(str(path),int(pos))

            
if __name__ == '__main__':

    root = Tk.Tk()
    root.wm_title("Welcome to Mass. Eye and Ear - Smart Facial Metrics Tool")
    __location__ = os.path.dirname(os.path.realpath(sys.argv[0]))
    root.iconbitmap(__location__ + '\\include\\meei_3WR_icon.ico')
    #first row
    ImageFol = Tk.Label(master = root, text = 'Image Folder:', width = 16, anchor=Tk.W)
    ImageFol.grid(row=0, column = 0, columnspan=1, sticky = Tk.W, padx = 4)
    
    TextPath = Tk.Entry(master = root, width = 70,)
    TextPath.grid(row=0, column = 1, columnspan=2, sticky = Tk.W, padx = 4, pady=4)
    
    broButton = Tk.Button(master = root, text = 'Browse Folder', width = 13, command=browse_file)
    broButton.grid(row=0, column = 3, columnspan=1, sticky = Tk.W, padx = 4, pady=4)
    
    #second row
    InitPos = Tk.Label(master = root, text = 'Initial Position:', width = 16, anchor=Tk.W)
    InitPos.grid(row=1, column = 0, columnspan=1, sticky = Tk.W, padx = 4)
    
    TextInitPos = Tk.Entry(master = root, width = 30)
    TextInitPos.insert(Tk.END, '0')
    TextInitPos.grid(row=1, column = 1, columnspan=1, sticky = Tk.W, padx = 4, pady=4)
    TextInitPos.bind('<Button-1>', clear_box)
    
    #third row
    StartButton = Tk.Button(master = root, text = 'Start', width = 13, command=start_process)
    StartButton.grid(row=2, column = 1, columnspan=1, sticky = Tk.W, padx = 4, pady=4)
    
    QuitButton = Tk.Button(master = root, text = 'Quit', width = 13, command = close_window)
    QuitButton.grid(row=2, column = 2, columnspan=1, sticky = Tk.W, padx = 4, pady=4)
    
    
    
    Tk.mainloop()