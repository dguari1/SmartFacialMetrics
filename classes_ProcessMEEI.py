# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 21:42:28 2017

@author: Diego L. Guarin -- diego_guarin at meei.harvard.edu
"""
import cv2

#file containing the classes using in the main program
class CoordinateStore:
    def __init__(self):
        self.points = []
        self.click = 0

    def select_point(self,event,x,y,flags,param):
            if event == cv2.EVENT_RBUTTONDOWN :
                #if right button down then flag=1 and store position
                self.click = 1
                self.points = [x,y]
                
            if event == cv2.EVENT_RBUTTONUP:
                #if right button up then flag=0
                self.click = 0
                
            if event == cv2.EVENT_LBUTTONDOWN :
                #if left button down then flag=2 and store position
                self.click = 2
                self.points = [x,y]
                
            if event == cv2.EVENT_LBUTTONUP :
                #if left button up then flag=0
                self.click = 0
                

                
class CoordinateStoreEye:
    def __init__(self):
        self.points = []
        self.click = 0
        
    def select_point(self,event,x,y,flags,param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points = [x,y]
            self.click = 1
            
        if event == cv2.EVENT_LBUTTONUP:
            self.click = 0
