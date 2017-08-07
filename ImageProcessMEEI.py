# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 16:12:15 2017

@author: Diego L. Guarin -- diego_guarin at meei.harvard.edu
"""

import cv2
import numpy as np
import sys
import argparse
from scipy.spatial.distance import cdist

import os


from dlib import get_frontal_face_detector
from dlib import shape_predictor
from dlib import rectangle

#import utilities
from utils_ProcessMEEI import GetPupil
from utils_ProcessMEEI import find_circle_from_points
from utils_ProcessMEEI import save_txt_file
from utils_ProcessMEEI import shape_to_np
from utils_ProcessMEEI import draw_on_picture
from utils_ProcessMEEI import estimate_lines
from utils_ProcessMEEI import compute_measurements

#import classes 
from classes_ProcessMEEI import CoordinateStoreEye
from classes_ProcessMEEI import CoordinateStore
#import time



#utility funtions (that use the classes)
def process_eye(EyeImage):
    
    temp_image=EyeImage.copy()
    #start the class to detect the clicks
    GetCoordsonClickEye = CoordinateStoreEye()
    #create a new cv2 window, and associate the class with the window
    cv2.namedWindow("EyeImage", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("EyeImage",GetCoordsonClickEye.select_point)
    cv2.imshow("EyeImage", temp_image)
    
    getEyes=np.empty([4, 2], dtype = int)
    #count the number of click, we need four clicks to select the eye
    clickCounter=0
    while(True):
        cv2.waitKey(1)
        
        if GetCoordsonClickEye.click == 1 :
            getEyes[clickCounter,:]= GetCoordsonClickEye.points
            cv2.circle(temp_image,
                (GetCoordsonClickEye.points[0],GetCoordsonClickEye.points[1]), 
                1, (0,0,255),1)           
            cv2.imshow("EyeImage", temp_image)
            clickCounter += 1
            cv2.waitKey(100)
            
        if clickCounter == 4:
            break
        
    #now that we have the four points around the pupil, compute the center and
    #radii of the pupil 
    circle = find_circle_from_points(getEyes[:,0],getEyes[:,1])   
    cv2.imshow("EyeImage", temp_image)
    cv2.destroyWindow("EyeImage")
    
    return circle


def process_image(file_to_read, path, image, shape, mod_rect):
    #this function takes care of do all the processing of a valid image
    
    
    #initialize the class to get coordinates on click event 
    GetCoordsonClick = CoordinateStore()
    
    #show image on window
    cv2.namedWindow("MainWindow", cv2.WINDOW_NORMAL)
    #associate the click class with the window
    cv2.setMouseCallback("MainWindow",GetCoordsonClick.select_point)
    cv2.imshow('MainWindow',image)       
    
    
    
    #make local copy of the image and the original points, these will be modified 
    #through the execution of this program 
    temp_image=image.copy()
    temp_shape=shape.copy()
    
    
    #locate pupil on left eye
    x_left = temp_shape[42,0]
    w_left = (temp_shape[45,0]-x_left)
    y_left = min(temp_shape[43,1],temp_shape[44,1])
    h_left = (max(temp_shape[46,1],temp_shape[47,1])-y_left)
    LeftEye = temp_image.copy()
    LeftEye = LeftEye[(y_left-5):(y_left+h_left+5),(x_left-5):(x_left+w_left+5)]
    selected_circle_left = GetPupil(LeftEye)
    selected_circle_left[0]=selected_circle_left[0]+x_left-5
    selected_circle_left[1]=selected_circle_left[1]+y_left-5
    #print a point in the center of the pupil and a circle in it
    
    
    #locate pupil on right eye
    x_right = temp_shape[36,0]
    w_right = (temp_shape[39,0]-x_right)
    y_right = min(temp_shape[37,1],temp_shape[38,1])
    h_right = (max(temp_shape[41,1],temp_shape[40,1])-y_right)
    RightEye = temp_image.copy()
    RightEye = RightEye[(y_right-5):(y_right+h_right+5),(x_right-5):(x_right+w_right+5)]
    selected_circle_right = GetPupil(RightEye)
    selected_circle_right[0]=selected_circle_right[0]+x_right-5
    selected_circle_right[1]=selected_circle_right[1]+y_right-5
    #print a point in the center of the pupil and a circle in it
    
    #print in the image   
    temp_image=draw_on_picture(temp_image, temp_shape, 
                               selected_circle_left, selected_circle_right)
    

    
    #this flag will indicates if we need to start modifying the points in the face
    ShouldModify = 0
    #this flag will indicates if a point was lifted from the face  
    IsPointLifted=0
    #this are the points that create a ruler in the face
    points = None
    while(True):
        cv2.imshow('MainWindow',temp_image)
        
        
        #check for user interaction 
        KeyPressed = cv2.waitKey(10) 
        if  KeyPressed & 0xFF == ord('q'):
            #if keypressed = q then quit
            break
        elif KeyPressed & 0xFF == ord('c'):
            #if keypressed = m then start modifying image
            ShouldModify = 1
            #print(ShouldModify)
        elif KeyPressed & 0xFF == ord('s'):
            #if keypressed = s then save results into txt file
            
            temp_image =compute_measurements(file_to_read, 
                  path, temp_shape,selected_circle_left, selected_circle_right,
                  temp_image)
            save_txt_file(file_to_read,path, temp_shape,mod_rect,
                          selected_circle_left,selected_circle_right)
            
            #update the image 
            temp_image = draw_on_picture(temp_image, temp_shape, 
                              selected_circle_left, selected_circle_right, 
                              points)
    
            #and show it
            #cv2.imshow('MainWindow',temp_image)
            #cv2.waitKey(100)
            ShouldModify = 0
        elif KeyPressed & 0xFF == ord('d'):
            #if keypressed = d then move to the next image
            break
        elif KeyPressed & 0xFF == ord('a'):
            #if keypressed = a then move to the previous image
            break
        elif KeyPressed & 0xFF == ord('r'):
            #if keypressed = r then draw a ruler lines in the face 
            #if the iris is found the drow some lines to divide the face
            if selected_circle_left[2] > 0 and selected_circle_right[2] > 0:
                
                #compute the line between eyes and a perpendicular line in the 
                #middle 
                points = estimate_lines(temp_image,
                                       selected_circle_left, 
                                       selected_circle_right)
                #update the image 
                temp_image = draw_on_picture(temp_image, temp_shape, 
                              selected_circle_left, selected_circle_right, points)
    
                #and show it
                #cv2.imshow('MainWindow',temp_image)
        elif KeyPressed & 0xFF == ord('g'):
            #the user wants to save a copy of the marked picture
            cv2.imwrite(path+'\\'+ file_to_read[:-4]+'_processed.jpg', temp_image)
                

            
    
        #if the user wants to modify the image then start doing that 
        if ShouldModify == 1:  
            #if no point is lifted and the user performs a left-click event, then 
            #lift the selected point
            if IsPointLifted == 0 and GetCoordsonClick.click == 1:
                #get current position of mouse and verify if is very close to one 
                #of the points printed in the image
                
                #start = time.time()
                #we will include the 68 landamaks and the center of both pupils for 
                #a total of 70 points to compare
                MousePosition=np.ones((70,2),dtype="int")
                MousePosition[:,0]=MousePosition[:,0]*GetCoordsonClick.points[0]
                MousePosition[:,1]=MousePosition[:,1]*GetCoordsonClick.points[1]
                
                #stacking the position of the 68 landmaks with the center of left
                #and right eye
                comp_temp_shape=np.append(temp_shape,
                    [selected_circle_left[0:2],selected_circle_right[0:2]],axis=0)
                
                distance=cdist(comp_temp_shape,  MousePosition)
                distance=distance[:,0]
                PointToModify = [i for i, j in enumerate(distance) if j <=3 ]
                
                #if the used clicked on one particular point then lift that point
                #from the face
                if PointToModify:               
                    #if by any chance more than one point is clicke (two very closed-by)
                    #points then we need to keep only one. 
                    PointToModify = PointToModify[0]
                    
                    #we need to separate click in the eye vs click in the face
                    if PointToModify >= 68:
                        #this is a click in the eye, the user wants to re-do the 
                        #pupil selection
                        if PointToModify == 68:
                            #user wants to modify the left eye
                            selected_circle_left = process_eye(LeftEye)
                            selected_circle_left[0]=selected_circle_left[0]+x_left-5
                            selected_circle_left[1]=selected_circle_left[1]+y_left-5
                            
                            #if there is a change in the ladmarks then save
                            #automatically
                            save_txt_file(file_to_read,path, temp_shape,mod_rect,selected_circle_left,selected_circle_right)

                        if PointToModify == 69:
                            #user wants to modify the right eye
                            selected_circle_right = process_eye(RightEye)
                            selected_circle_right[0]=selected_circle_right[0]+x_right-5
                            selected_circle_right[1]=selected_circle_right[1]+y_right-5  
                            
                            #if there is a change in the ladmarks then save
                            #automatically
                            save_txt_file(file_to_read,path, temp_shape,mod_rect,selected_circle_left,selected_circle_right)

                    elif PointToModify <= 67:
                        #this is a click in one of the landmarks, the user wants
                        #to modify the landmark location                
                        #we set the position of the selected landmark to negative
                        #values so that it will be removed from the image
                        temp_shape[PointToModify]=[-1,-1]
                        #we need to inform to the outer loop that a point was lifted 
                        #from the face
                        IsPointLifted = 1
                        
                    
                    #now we re-draw the image with the new pupil or with the 
                    #new point lifted 
                    temp_image=image.copy()
                    temp_image = draw_on_picture(temp_image, temp_shape, 
                                      selected_circle_left, selected_circle_right,points)
                    #cv2.imshow('MainWindow',temp_image)

                    #cv2.waitKey(50)    
                    
    
            #if a point is lifted and the user performs a right-click event, then 
            #relocate the point that was lifted 
            if IsPointLifted == 1 and GetCoordsonClick.click == 2 :
                #update the points in the face and print
                temp_image=image.copy()
                temp_shape[PointToModify]=GetCoordsonClick.points
                #if the modified point was in the left eye, then update the 
                #pupil acquisition 
                if 42 <= PointToModify <= 47:
                    #locate pupil on left eye
                    x_left = temp_shape[42,0]
                    w_left = (temp_shape[45,0]-x_left)
                    y_left = min(temp_shape[43,1],temp_shape[44,1])
                    h_left = (max(temp_shape[46,1],temp_shape[47,1])-y_left)
                    LeftEye = temp_image.copy()
                    LeftEye = LeftEye[(y_left-5):(y_left+h_left+5),(x_left-5):(x_left+w_left+5)]
                    selected_circle_left = GetPupil(LeftEye)
                    selected_circle_left[0]=selected_circle_left[0]+x_left-5
                    selected_circle_left[1]=selected_circle_left[1]+y_left-5
                elif 36 <= PointToModify <= 41:
                    #locate pupil on right eye
                    x_right = temp_shape[36,0]
                    w_right = (temp_shape[39,0]-x_right)
                    y_right = min(temp_shape[37,1],temp_shape[38,1])
                    h_right = (max(temp_shape[41,1],temp_shape[40,1])-y_right)
                    RightEye = temp_image.copy()
                    RightEye = RightEye[(y_right-5):(y_right+h_right+5),(x_right-5):(x_right+w_right+5)]
                    selected_circle_right = GetPupil(RightEye)
                    selected_circle_right[0]=selected_circle_right[0]+x_right-5
                    selected_circle_right[1]=selected_circle_right[1]+y_right-5
                    
    
                #if there is a change in the ladmarks then save
                #automatically
                save_txt_file(file_to_read,path, temp_shape,mod_rect,selected_circle_left,selected_circle_right)

                #update the figure
                temp_image = draw_on_picture(temp_image, temp_shape, 
                              selected_circle_left, selected_circle_right,points)
    
                #and show it
                #cv2.imshow('MainWindow',temp_image)
                
                #now that the point is relocated go back to the state where no
                #point is lifted
                IsPointLifted = 0
                #cv2.waitKey(50)
            
        #cv2.waitKey(50) 

    return KeyPressed
    
    

def main(path,k):
    
    #load face detector and landmark predictor 
    detector = get_frontal_face_detector()
    #find the path were the script is saved
    __location__ = os.path.dirname(os.path.realpath(sys.argv[0]))
    #verify if the model exists
    if os.path.isfile(__location__ + '\\shape_predictor_68_face_landmarks.dat'):
        #if it exists then load it 
        predictor = shape_predictor(__location__ + '\\shape_predictor_68_face_landmarks.dat')
    else:
        sys.exit('No landmark model avaliale')    
    
    
    #read files in forder
    #path=r'C:\Users\diego\Dropbox (NRP)\Data\DataBase\Patients'
    #path=r'C:\Users\diego\Documents\Python Scripts\onclickevent'
    files = os.listdir(path)
    ext=('.png', '.jpg', '.jpeg', '.bmp', '.PNG', '.JPG', '.JPEG', '.BMP')
    files_to_read = [i for i in files if i.endswith(tuple(ext))]
    if len(files_to_read) == 0:
        sys.exit('No valid images in directory. Valid formats include .png, .jpg, .jpge and .bmp') 
    
    #path='C:/Users/diego/Dropbox (NRP)/Data/DataBase/Patients/'
    #files_jpg='Slide2.png'
    
    #k=0
    while k < (len(files_to_read)):
    #for k in range(0,len(files_to_read)):
        #load image and convert it to gray
        file_to_read=files_to_read[k]
        image=cv2.imread(path + '\\'+file_to_read)
        
        #the image is in memory, so let's create a window and display it
        cv2.namedWindow("MainWindow", cv2.WINDOW_GUI_NORMAL + cv2.WINDOW_KEEPRATIO)
        #cv2.WINDOW_GUI_NORMAL
        #cv2.setMouseCallback("MainWindow",GetCoordsonClick.select_point)
        cv2.imshow('MainWindow',image)
        #if (cv2.waitKey(1)&0xff) == 27: break
        
        #now we start the image processing component......
        gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        
        #resize image to speed up face dectection
        height, width = gray.shape[:2]  
        newWidth=200
        ScalingFactor=width/newWidth
        newHeight=int(height/ScalingFactor)
        smallImage=cv2.resize(gray, (newWidth, newHeight), interpolation=cv2.INTER_AREA)
    
        #detect face in image using dlib.get_frontal_face_detector()
        rects = detector(smallImage,1)
        
        if len(rects) == 0:
            #there is no face, the user should move on or quit 
            cv2.putText(image,"No face detected", 
                        (int(height/5),int(width/2)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255),2)
            while(True):
                #show image
                cv2.imshow('MainWindow',image)
                #and wait for user input 
                KeyPressed = cv2.waitKey(10) 
                if  KeyPressed & 0xFF == ord('q'):
                    #if keypressed = q then quit
                    cv2.destroyAllWindows()
                    return 0
                    #sys.exit('exit')
                elif KeyPressed & 0xFF == ord('d'):
                    #if keypressed = d then move to the next image                    
                    k += 1
                    break
                elif KeyPressed & 0xFF == ord('a'):
                    #if keypressed = a then move to the previous image 
                    k -= 1
                    #if in the first image then do nothing  
                    if k<0:
                        k=0
                    break
        elif len(rects) > 1:
            #there are multiple faces, the user should move on or quit 
            cv2.putText(image,"Multiple faces detected", 
                        (int(height/5),int(width/2)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255),2)
            while(True):
                #show image
                cv2.imshow('MainWindow',image)
                #and wait for user input 
                KeyPressed = cv2.waitKey(10) 
                if  KeyPressed & 0xFF == ord('q'):
                    #if keypressed = q then quit
                    cv2.destroyAllWindows()
                    return 0
                    #sys.exit('exit')
                elif KeyPressed & 0xFF == ord('d'):
                    #if keypressed = d then move to the next image
                    k += 1
                    break
                elif KeyPressed & 0xFF == ord('a'):
                    #if keypressed = a then move to the previous image
                    k -= 1
                    #if in the first image then do nothing
                    if k<0:
                        k=0
                    break
        elif len(rects) == 1:
            #now we have only one face in the image
            #function to obtain facial landmarks using dlib 
            #given an image and a face
            #rectangle
            for (i, rect) in enumerate(rects):
                # determine the facial landmarks for the face region, then
                # convert the facial landmark (x, y)-coordinates to a NumPy array
    
                #adjust face position using the scaling factor
                mod_rect=rectangle(
                        left=int(rect.left() * ScalingFactor), 
                        top=int(rect.top() * ScalingFactor), 
                        right=int(rect.right() * ScalingFactor), 
                        bottom=int(rect.bottom() * ScalingFactor))
       
                #predict facial landmarks 
                shape = predictor(image, mod_rect)   
            
                #transform shape object to np.matrix type
                shape = shape_to_np(shape)

            #this function takes care of do all the processing of a valid image
            #it doesn't return anything 
            KeyPressed = process_image(file_to_read, path, image, shape, mod_rect)
            
            if  KeyPressed & 0xFF == ord('q'):
                #if keypressed = q then quit
                cv2.destroyAllWindows()
                return 0
                #sys.exit('exit')
            elif KeyPressed & 0xFF == ord('d'):
                #if keypressed = d then move to the next image
                k += 1
            elif KeyPressed & 0xFF == ord('a'):
                #if keypressed = a then move to the previous image
                k -= 1
                #if in the first image then do nothing
                if k<0:
                    k=0

        
    cv2.destroyAllWindows()
    return 0
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--folder", type=str, required=True)
    parser.add_argument("-p", "--position", type=int, default=0)
    args = parser.parse_args() 
    path=args.folder   
    k=args.position
    
    
    main(path,k)