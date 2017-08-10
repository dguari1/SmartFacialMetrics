# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 13:57:28 2017

@author: Diego L. Guarin -- diego_guarin at meei.harvard.edu
"""

import cv2
import numpy as np
from scipy import linalg
import os
from tabulate import tabulate


  

def shape_to_np(shape,dtype="int"):
    #function to transform the results provided by dlib into usable np arrays
    coords=np.zeros((68,2),dtype=dtype)
    #the figure was reduced by 10 to speed things-up
    for i in range(0,68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
        
    return coords




def draw_on_picture(image, shape, circle_left, circle_right, points=None):
    #function to draw on the image 
    
    h,w,_=image.shape
    #print(int(w/160))
    
    #if requested, then draw a two lines to divide the face
    if points is not None:
        cv2.line(image,points[0],points[1],(0,255,0),1)        
        cv2.line(image,points[2],points[3],(0,255,0),1)
        cv2.line(image,points[4],points[5],(0,255,0),1)

    #draw 68 landmark points
    aux=1
    for (x,y) in shape:
        if x>0:
            mark_size=int(w/180)
            if mark_size>4: mark_size=4
            cv2.circle(image, (x,y), mark_size , (0,0,255),-1)
            cv2.putText(image, str(aux), (x-2,y-2), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0,0,0), 1)
            #           
        aux +=1
    
    #draw left pupil
    if circle_left[2]>0:
        cv2.circle(image, 
               tuple([int(circle_left[0]),
               int(circle_left[1])]),
               int(circle_left[2]),(0,255,0),1)
        cv2.circle(image, 
               tuple([int(circle_left[0]),
               int(circle_left[1])]),
               int(w/200),(0,255,0),-1)
    
    #draw right pupil
    if circle_right[2]>0:
        cv2.circle(image, 
               tuple([int(circle_right[0]),
               int(circle_right[1])]),
               int(circle_right[2]),(0,255,0),1)
        cv2.circle(image, 
               tuple([int(circle_right[0]),
               int(circle_right[1])]),
               int(w/200),(0,255,0),-1)
    
    return image

def GetPupil(InputImage):
    
    #this function appplies a modified Daugman procedure for iris detection.
    #See 'How Iris Recognition Works, Jhon Dougman - IEEE Transactions on 
    #circuits and systems for video technology, January 2004'
    
    #get dimension of image 
    h_eye, w_eye, d_eye = InputImage.shape
    
    #this is the variable that will be return after processing
    circle=[]
    
    #verify that it is a color image
    if d_eye < 3:
        print('Pupil cannot be detected -- Color image is required')
        #circle=[int(w_eye/2), int(h_eye/2), int(w_eye/4)]
        circle=[-1,-1,-1]
        return circle
        
    #verify that the eye is open   
    if w_eye/h_eye > 3.2:
        print('Pupil cannot be detected -- Eye is closed')
        #circle=[int(w_eye/2), int(h_eye/2), int(w_eye/4)]
        circle=[-1,-1,-1]
        return circle
    
    #reduce brightness to help with light-colored eyes
    InputImage=np.array(InputImage*0.75+0, dtype=InputImage.dtype)
    
    #split image into its different color channels 
    b,g,r = cv2.split(InputImage)
    
    #and create a new gray-image combining the blue and green channels, this 
    #will help to differentiate between the iris and sclera 
    #the function cv2.add guarantees that the resulting image has values 
    #between 0 (white) and 255 (black)
    bg = cv2.add(b,g)
    #filter the image to smooth the borders
    bg = cv2.GaussianBlur(bg,(3,3),0)
    
    #we assume that the radii of the iris is between 1/5.5 and 1/3.5 times the eye 
    #width (this value was obtained after measuring multiple eye images, it only
    #works if the eye section was obtained via dlib)
    Rmin=int(w_eye/5.5)
    Rmax=int(w_eye/3.5)
    radius=range(Rmin,Rmax+1)
    
    result_value=np.zeros(bg.shape, dtype=float)
    result_index_ratio=np.zeros(bg.shape, dtype=bg.dtype)
    mask = np.zeros(bg.shape, dtype=bg.dtype)
    
    #apply the Dougnman's procedure for iris detection. In this case I modify the 
    #procedure instead of use a full circunference it only uses 1/5 of 
    #a circunference. The procedure uses a circle between -35deg-0deg and 
    #180deg-215deg if the center beeing analized is located in the top half of the 
    #eye image, and a circle between 0deg-35deg and 145deg-180deg if the center 
    #beeing analized is located in the bottom half of the eye image
    
    possible_x=range(Rmin,w_eye-Rmin)
    possible_y=range(0,h_eye)
    for x in possible_x:
        for y in possible_y:  
                      
            intensity=[]
            for r in radius:
                
                if y>=int(h_eye/2):
                    temp_mask=mask.copy()   
                    #cv2.circle(temp_mask,(x,y),r,(255,255,255),1)
                    cv2.ellipse(temp_mask, (x,y), (r,r), 0, -35, 0, (255,255,255),1)
                    cv2.ellipse(temp_mask, (x,y), (r,r), 0, 180, 215, (255,255,255),1)
                    processed = cv2.bitwise_and(bg,temp_mask)
                    intensity.append(cv2.sumElems(processed)[0]/(2*3.141516*r))
                
                else:
                    temp_mask=mask.copy()   
                    #cv2.circle(temp_mask,(x,y),r,(255,255,255),1)
                    cv2.ellipse(temp_mask, (x,y), (r,r), 0, 0, 35, (255,255,255),1)
                    cv2.ellipse(temp_mask, (x,y), (r,r), 0, 145, 180, (255,255,255),1)
                    processed = cv2.bitwise_and(bg,temp_mask)
                    intensity.append(cv2.sumElems(processed)[0]/(2*3.141516*r))                
    
    
            diff_vector=np.diff(intensity)
            max_value=max(diff_vector)
            max_index = [i for i, j in enumerate(diff_vector) if j == max_value]   
            result_value[y,x]=max_value
            result_index_ratio[y,x]=max_index[0]
        
    
    
    #the results are filtered by a Gaussian filter, as suggested by Daugman
    result_value=cv2.GaussianBlur(result_value,(7,7),0)
    
    
    
    #now we need to find the center and radii that show the largest change in 
    #intensity    
    matrix = result_value
    needle = np.max(matrix)
    
    matrix_dim = w_eye
    item_index = 0
    for row in matrix:
        for i in row:
            if i == needle:
                break
            item_index += 1
        if i == needle:
            break
    
    #this is the center and radii of the selected circle
    c_y_det=int(item_index / matrix_dim) 
    c_x_det=item_index % matrix_dim
    r_det=radius[result_index_ratio[c_y_det,c_x_det]]
    
    circle=[c_x_det,c_y_det,r_det]   
    
    return circle 


def find_circle_from_points(x,y):
    #this function finds the center and radius of a circle from a set of points 
    #in the circle. x and y are the coordinates of the points. 
    
    # coordinates of the barycenter
    x_m = np.mean(x)
    y_m = np.mean(y)

    # calculation of the reduced coordinates
    u = x - x_m
    v = y - y_m

    # linear system defining the center (uc, vc) in reduced coordinates:
    #    Suu * uc +  Suv * vc = (Suuu + Suvv)/2
    #    Suv * uc +  Svv * vc = (Suuv + Svvv)/2
    Suv  = sum(u*v)
    Suu  = sum(u**2)
    Svv  = sum(v**2)
    Suuv = sum(u**2 * v)
    Suvv = sum(u * v**2)
    Suuu = sum(u**3)
    Svvv = sum(v**3)

    # Solving the linear system
    A = np.array([ [ Suu, Suv ], [Suv, Svv]])
    B = np.array([ Suuu + Suvv, Svvv + Suuv ])/2.0
    uc, vc = linalg.solve(A,B)

    xc_1 = x_m + uc
    yc_1 = y_m + vc

    # Calcul des distances au centre (xc_1, yc_1)
    Ri_1     = np.sqrt((x-xc_1)**2 + (y-yc_1)**2)
    R_1      = np.mean(Ri_1)

    
    circle=[int(xc_1),int(yc_1),int(R_1)]
    #circle.append((int(xc_1),int(yc_1),int(R_1)))
    
    return circle

def estimate_lines(InputImage,circle_left, circle_right):
    #function to estimate the line that connects the center of the eyes and a 
    #new, perpendicular line in the middle point.
    
    h, w, _ = InputImage.shape
    
    x_1=circle_right[0]
    y_1=circle_right[1]
    
    x_2=circle_left[0]
    y_2=circle_left[1]
    
    #find the point in the middle of the line
    x_m=((x_2-x_1)/2)+x_1
    m=(y_2-y_1)/(x_2-x_1)   
    y_m=(y_1+m*(x_m-x_1))
    
    x_m=int(round(x_m,0))
    y_m=int(round(y_m,0))
    angle=np.arctan(m)+np.pi/2
    
    
    x_p1=int(round(x_m+0.75*h*np.cos(angle)))
    y_p1=int(round(y_m+0.75*h*np.sin(angle)))
    
    x_p2=int(round(x_m-0.75*h*np.cos(angle)))
    y_p2=int(round(y_m-0.75*h*np.sin(angle)))     
    
    
    points=[(x_1,y_1),(x_2,y_2),(x_m,y_m),(x_p1,y_p1),(x_m,y_m),(x_p2, y_p2)]
    
    return points


def compute_measurements(file, path, shape,circle_left, circle_right, image):
    #function that computes diverse measurements in the face, all the distance
    #measurements are normalized with respecto the iris radii 
    
    h, w, _ = image.shape
                        
    #compute the desired measurements 
    radii_left=circle_left[2]
    radii_right=circle_right[2]
    #compute the average pupul radii, this will be used to normalize all the measures
    radii=(radii_left+radii_right)/2
    
    #points in the center of th eyes, this infor will be used throughout this 
    #function 
    #right
    x_1=circle_right[0]
    y_1=circle_right[1]
    #left
    x_2=circle_left[0]
    y_2=circle_left[1]    
    #and slope of the line connecting those points 
    m=(y_2-y_1)/(x_2-x_1)
    
    
           
    
    #MOUTH MEASUREMENTS     
    #find all the necesary points to create parallel lines 

    #computing angles and distance in left smile
    x_l=shape[54,0]
    y_l=shape[54,1]
    x_c=shape[57,0]
    y_c=shape[57,1]
    
    x_par=x_l+0.5*w
    y_par=int(round(y_c+m*(x_par-x_c),0))
    #cv2.line(image,(x_c,y_c),(x_par,y_par),(0,255,0),1 )
    
    y_per=y_c+0.5*h
    x_per=int(round(x_l-m*(y_per-y_l),0))


    x_cross, y_cross = line_intersection(((x_c,y_c),(x_par,y_par)), 
                                         ((x_l,y_l),(x_per,y_per)))
    x_cross=int(round(x_cross,0))
    y_cross=int(round(y_cross,0))
    
    polypoints_left=np.array([tuple([x_c,y_c]),tuple([x_l,y_l]),tuple([x_cross,y_cross])], np.int32)
    #draw this localy, it will desapear if user modify landmarks  
    cv2.polylines(image,[polypoints_left],1,(255,0,0),1)
    
    
    opposite_left = np.sqrt((y_l-y_cross)**2 + (x_l-x_cross)**2)
    adjacent_left = np.sqrt((y_c-y_cross)**2 + (x_c-x_cross)**2)
    hypothenuse_left = np.sqrt(opposite_left**2 + adjacent_left**2)
    #becuase of the way we are constructing these triangels, we loose all the 
    #information about quadrants. So, we need to manually separate the different 
    #cases : 1) lower lip lower that mouth corner (positive angle), 
    #2) lower lip higher than mouth corner (negative angle), 
    #3) lower lip same level as mouth corner (zero angle)
    if y_l < y_c: #case 1
        smile_angle_left = (np.arctan(opposite_left/adjacent_left)+np.arctan(m)+np.pi/2)*(180/np.pi)   
        cv2.ellipse(image, (x_c,y_c), (20,20), np.arctan(m)*(180/np.pi),90, -np.arctan(opposite_left/adjacent_left)*(180/np.pi),(0,0,0),1)

    elif y_l > y_c: #case 2
        smile_angle_left = (-np.arctan(opposite_left/adjacent_left)+np.arctan(m)+np.pi/2)*(180/np.pi)
        cv2.ellipse(image, (x_c,y_c), (20,20), np.arctan(m)*(180/np.pi),90, np.arctan(opposite_left/adjacent_left)*(180/np.pi),(0,0,0),1)

    
    elif y_l == y_c: #case 3
        smile_angle_left = (np.arctan(m)+np.pi/2)*(180/np.pi)
        #draw this localy, it will desapear if user modify landmarks 
        cv2.line(image,(x_l,y_l),(x_c,y_c),(0,0,0),1 )

    
    
    #computing angles and distance in right smile
    x_r=shape[48,0]
    y_r=shape[48,1]
    x_c=shape[57,0]
    y_c=shape[57,1]
    
    x_par=x_r+0.5*h
    y_par=int(round(y_c+m*(x_par-x_c),0))
    
    
    
    y_per=y_c+0.5*w
    x_per=int(round(x_r-m*(y_per-y_r),0))

    x_cross, y_cross = line_intersection(((x_c,y_c),(x_par,y_par)), 
                                         ((x_r,y_r),(x_per,y_per)))
    x_cross=int(round(x_cross,0))
    y_cross=int(round(y_cross,0))
    
    polypoints_right=np.array([tuple([x_c,y_c]),tuple([x_r,y_r]),tuple([x_cross,y_cross])], np.int32)
    #draw this localy, it will desapear if user modify landmarks  
    cv2.polylines(image,[polypoints_right],1,(255,0,0),1)
    
    opposite_right = np.sqrt((y_r-y_cross)**2 + (x_r-x_cross)**2)
    adjacent_right = np.sqrt((y_c-y_cross)**2 + (x_c-x_cross)**2)
    hypothenuse_right = np.sqrt(opposite_right**2 + adjacent_right**2)
    #becuase of the way we are constructing these triangels, we loose all the 
    #information about quadrants. So, we need to manually separate the different 
    #cases : 1) lower lip lower that mouth corner (positive angle), 
    #2) lower lip higher than mouth corner (negative angle), 
    #3) lower lip same level as mouth corner (zero angle)
    if y_r < y_c: #case 1
        smile_angle_right = (np.arctan(opposite_right/adjacent_right)+np.arctan(m)+np.pi/2)*(180/np.pi)
        #draw this localy, it will desapear if user modify landmarks           
        cv2.ellipse(image, (x_c,y_c), (20,20), np.arctan(m)*(180/np.pi),90,180+np.arctan(opposite_right/adjacent_right)*(180/np.pi),(0,0,255),1)
    
    elif y_r > y_c: #case 2
        smile_angle_right = (-np.arctan(opposite_right/adjacent_right)+np.arctan(m)+np.pi/2)*(180/np.pi)
        #draw this localy, it will desapear if user modify landmarks   
        cv2.ellipse(image, (x_c,y_c), (20,20), np.arctan(m)*(180/np.pi),90,180-np.arctan(opposite_right/adjacent_right)*(180/np.pi),(0,0,255),1)
    
    elif y_r == y_c: #case 3
        smile_angle_right = (np.arctan(m)+np.pi/2)*(180/np.pi)
        #draw this localy, it will desapear if user modify landmarks   
        cv2.line(image,(x_r,y_r),(x_c,y_c),(0,0,0),1 )


    #Mouth extension 
    Mouth_extension = np.sqrt((x_l-x_r)**2 + (y_l-y_c)**2)
    
    #Philthral deviation
    x_philtrum = shape[51,0]
    y_philtrum = shape[51,1]
    
    y_cross_philtrum=y_philtrum
    x_cross_philtrum=int(round(x_c+m*(y_cross_philtrum-x_c),0))
    Philthral_deviation=np.sqrt((x_philtrum-x_cross_philtrum)**2 + (y_philtrum-y_cross_philtrum)**2)
    
    
    
    #point in the  middle of the eyes, this will be used as reference to 
    #compute all the measurements 
    x_m=((x_2-x_1)/2)+x_1
    m=(y_2-y_1)/(x_2-x_1)   
    y_m=(y_1+m*(x_m-x_1))
    
    x_m=int(round(x_m,0))
    y_m=int(round(y_m,0))
    
    
    
    #Oral commissure malposition

    
    #left side
    x_par_left=x_l-0.5*w
    y_par_left=y_l+m*(x_par_left-x_l)  
    x_cross_left, y_cross_left = line_intersection(((x_c,y_c),(x_m,y_m)), 
                                         ((x_l,y_l),(x_par_left,y_par_left)))
#    ##some lines to verify what i did 
#    x_par_left=int(round(x_par_left,0))
#    y_par_left=int(round(y_par_left,0))
#    x_cross_left=int(round(x_cross_left,0))
#    y_cross_left=int(round(y_cross_left,0))
#    cv2.line(image,(x_l,y_l),(x_par_left,y_par_left),(0,0,0),1)
    
    #right side
    x_par_right=x_r+0.5*w
    y_par_right=y_r+m*(x_par_right-x_r)
  
    x_cross_right, y_cross_right = line_intersection(((x_c,y_c),(x_m,y_m)), 
                                         ((x_r,y_r),(x_par_right,y_par_right)))
#    ##some lines to verify what i did 
#    x_par_right=int(round(x_par_right,0))
#    y_par_right=int(round(y_par_right,0))
#    x_cross_right=int(round(x_cross_right,0))
#    y_cross_right=int(round(y_cross_right,0))    
#    cv2.line(image,(x_r,y_r),(x_par_right,y_par_right),(0,0,0),1)
    
    Oral_commissure_malposition=np.sqrt((x_cross_right-x_cross_left)**2 + 
                                       (y_cross_right-y_cross_left)**2 )
    
    
    
    
    #NOSE MEASUREMENTS 
    #nose deviation 
    x_nose = shape[33,0]
    y_nose = shape[33,1]
    
    y_cross_nose=y_nose
    x_cross_nose=int(round(x_c+m*(y_cross_nose-x_c),0))
    Nose_deviation=np.sqrt((x_nose-x_cross_nose)**2 + (y_nose-y_cross_nose)**2)
    
    #EYE MEASUREMENTS 
    ##!!! This can be improved in future version by fitting a curve to the eye!!!
    #left - superior lid malposition
    if shape[43,1] <= shape[44,1]:
        x_upper_left=shape[43,0]
        y_upper_left=shape[43,1]
    else:
        x_upper_left=shape[44,0]
        y_upper_left=shape[44,1]
        
    x_par_left=x_upper_left-0.5*w
    y_par_left=y_upper_left+m*(x_par_left-x_upper_left)  
    x_cross_left, y_cross_left = line_intersection(((x_c,y_c),(x_m,y_m)), 
                                         ((x_upper_left,y_upper_left),(x_par_left,y_par_left)))
    
    Sup_lid_malposition_left = np.sqrt((x_cross_left-x_m)**2 + (y_cross_left-y_m)**2)
#    ##some lines to verify what i did 
#    x_par_left=int(round(x_par_left,0))
#    y_par_left=int(round(y_par_left,0))
#    cv2.line(image,(x_upper_left,y_upper_left),(x_par_left,y_par_left),(0,0,0),1)



    #Right - superior lid malposition
    if shape[37,1] <= shape[38,1]:
        x_upper_right=shape[37,0]
        y_upper_right=shape[37,1]
    else:
        x_upper_right=shape[38,0]
        y_upper_right=shape[38,1]
        
    x_par_right=x_upper_right+0.5*w
    y_par_right=y_upper_right+m*(x_par_right-x_upper_right)  
    x_cross_right, y_cross_right = line_intersection(((x_c,y_c),(x_m,y_m)), 
                                         ((x_upper_right,y_upper_right),(x_par_right,y_par_right)))
    
    Sup_lid_malposition_right = np.sqrt((x_cross_right-x_m)**2 + (y_cross_right-y_m)**2)
    
    #superior lid ptosis
    Sup_lid_ptosis = np.sqrt((x_cross_right-x_cross_left)**2 + (y_cross_right-y_cross_left)**2)    


   
    #left - inferior lid malposition
    if shape[46,1] >= shape[47,1]:
        x_lower_left=shape[46,0]
        y_lower_left=shape[46,1]
    else:
        x_lower_left=shape[47,0]
        y_lower_left=shape[47,1]
        
    x_par_left=x_lower_left-0.5*w
    y_par_left=y_lower_left+m*(x_par_left-x_lower_left)  
    x_cross_left, y_cross_left = line_intersection(((x_c,y_c),(x_m,y_m)), 
                                         ((x_lower_left,y_lower_left),(x_par_left,y_par_left)))
    
    Inf_lid_malposition_left = np.sqrt((x_cross_left-x_m)**2 + (y_cross_left-y_m)**2)  
    
   
    #Right - inferior lid malposition
    if shape[40,1] >= shape[41,1]:
        x_lower_right=shape[40,0]
        y_lower_right=shape[40,1]
    else:
        x_lower_right=shape[41,0]
        y_lower_right=shape[41,1]
        
    x_par_right=x_lower_right+0.5*w
    y_par_right=y_lower_right+m*(x_par_right-x_lower_right)  
    x_cross_right, y_cross_right = line_intersection(((x_c,y_c),(x_m,y_m)), 
                                         ((x_lower_right,y_lower_right),(x_par_right,y_par_right)))
    
    Inf_lid_malposition_right = np.sqrt((x_cross_right-x_m)**2 + (y_cross_right-y_m)**2)  
    
    #inferior lid ptosis
    Inf_lid_ptosis = np.sqrt((x_cross_right-x_cross_left)**2 + (y_cross_right-y_cross_left)**2)  
    
    
    
    #BROWN MEASUREMENTS 
    #left - brown 
    x_brown_left = x_2
    x_brown_left, y_brown_left = get_eyebrown(shape[23:28], x_brown_left)
    x_par_left=x_brown_left-0.5*w
    y_par_left=y_brown_left+m*(x_par_left-x_brown_left)  
    x_cross_left, y_cross_left = line_intersection(((x_c,y_c),(x_m,y_m)), 
                                         ((x_brown_left,y_brown_left),(x_par_left,y_par_left)))
    
    Brown_elevation_left = np.sqrt((x_cross_left-x_m)**2 + (y_cross_left-y_m)**2)  
        
    
    #Right - brown 
    x_brown_right = x_1
    x_brown_right, y_brown_right = get_eyebrown(shape[18:23], x_brown_right)
    x_par_right=x_brown_right+0.5*w
    y_par_right=y_brown_right+m*(x_par_right-x_brown_right)  
    x_cross_right, y_cross_right = line_intersection(((x_c,y_c),(x_m,y_m)), 
                                         ((x_brown_right,y_brown_right),(x_par_right,y_par_right)))
    
    Brown_elevation_right = np.sqrt((x_cross_right-x_m)**2 + (y_cross_right-y_m)**2) 
    
    
    Brown_ptosis=np.sqrt((x_cross_left-x_cross_right)**2 + (y_cross_left-y_cross_right)**2)
    
    
    
    
    
    #write the information in a table 
    #if threre is a file containing results -- deleit it
    if os.path.isfile(path+'\\'+file[:-4]+'_measurements.txt'):
        os.remove(path+'\\'+file[:-4]+'_measurements.txt')
        
    with open(path+'\\'+file[:-4]+'_measurements.txt','a') as f: 
        f.write(tabulate([
        ['Mouth commissure to lower lip', round(hypothenuse_left/radii,3), round(hypothenuse_right/radii,3), ''], 
        ['Mouth commissure angle [deg]', round(smile_angle_left,2), round(smile_angle_right,3), ''], 
        ['Mouth extension', '' , '' , round(Mouth_extension/radii,3) ],
        ['Oral commissure malposition', '' , '' , round(Oral_commissure_malposition/radii,3) ],
        ['Philthral deviation', '' , '',round(Philthral_deviation/radii,3)],
        ['Nose deviation', '' , '',round(Nose_deviation/radii,3)],
        ['Superior lid malposition', round(Sup_lid_malposition_left/radii,3), round(Sup_lid_malposition_right/radii,3), ''],
        ['Superior lid ptosis', '', '', round(Sup_lid_ptosis/radii,3)],
        ['Inferior lid malposition', round(Inf_lid_malposition_left/radii,3), round(Inf_lid_malposition_right/radii,3), ''],
        ['Inferior lid ptosis', '', '', round(Inf_lid_ptosis/radii,3)],
        ['Brown elevation', round(Brown_elevation_left/radii,3), round(Brown_elevation_right/radii,3), ''],
        ['Brown ptosis', '', '', round(Brown_ptosis/radii,3)]
        ],
                    headers=['', 'Left' , 'Right', 'Global']))
        
        
        f.write('\n \n \nNote: All distances where normalized with respect to the iris radius')
        
        
    return image



def get_eyebrown(pts, new_x):
    #function that fits a third order polynomial to a group of 5 points that 
    #define the eyebrown in dlib. I then predicts the value of y for a new x
    y=np.array([[pts[0,1]],[pts[1,1]],[pts[2,1]],[pts[3,1]],[pts[4,1]]])
    X=np.array([[1, pts[0,0], pts[0,0]**2, pts[0,0]**3], 
                [1, pts[1,0], pts[1,0]**2, pts[1,0]**3],
                [1, pts[2,0], pts[2,0]**2, pts[2,0]**3],
                [1, pts[3,0], pts[3,0]**2, pts[3,0]**3],
                [1, pts[4,0], pts[4,0]**2, pts[4,0]**3]])
    params = np.linalg.lstsq(X, y)[0]
    new_y=params[0]+params[1]*new_x+ params[2]*(new_x**2)+ params[3]*(new_x**3)
    new_y=new_y[0]
    
    return new_x,new_y

def line_intersection(line1, line2):
    #comput the point where two line segments intersect 
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1]) #Typo was here

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
       raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y

def get_info_from_txt(file):
    shape=np.zeros((68,2),dtype=int)
    left_pupil = np.zeros((1,3),dtype=int)
    right_pupil = np.zeros((1,3),dtype=int)
    
    cont_landmarks = 0
    get_landmarks = 0
    
    get_leftpupil = 0
    cont_leftpupil = 0
    
    get_rightpupil = 0
    cont_rightpupil = 0
    with open(file, 'rt') as file:
        for i,line in enumerate(file):    
            if i == 4:    
                get_landmarks=1
            if i == 72:
                get_landmarks=0
                
            if get_landmarks == 1:
                temp=(line[:]).split(',')
                shape[cont_landmarks,0]=int(temp[0])
                shape[cont_landmarks,1]=int(temp[1])
                cont_landmarks += 1
                
            if i == 74:
                get_leftpupil = 1
            if i == 77:
                get_leftpupil = 0
                
            if get_leftpupil == 1:
                left_pupil[0,cont_leftpupil]=int(line[:])
                cont_leftpupil += 1
    
            if i == 79:
                get_rightpupil = 1
            if i == 82:
                get_rightpupil = 0
                
            if get_rightpupil == 1:            
                right_pupil[0,cont_rightpupil]=int(line[:])
                cont_rightpupil += 1      
                
    return shape, [left_pupil[0,0],left_pupil[0,1],left_pupil[0,2]],[right_pupil[0,0],right_pupil[0,1],right_pupil[0,2]]


def save_txt_file(file,path,shape,mod_rect,circle_left,circle_right):
    
    #create temporaty files with the information from the landamarks and 
    #both eyes
    #this piece is a bit weird, it creates three temporary files that are used
    #to create the final file, these files are then eliminated. This is
    #simplest (and easiest) way that i found to do this
    np.savetxt(path+'\\'+'temp_shape.txt', shape, delimiter=',',
                   fmt='%i', newline='\r\n')
    
    np.savetxt(path+'\\'+'temp_circle_left.txt', circle_left, delimiter=',',
                   fmt='%i', newline='\r\n')
    
    np.savetxt(path+'\\'+'temp_circle_right.txt', circle_right, delimiter=',',
                   fmt='%i', newline='\r\n')
        
    #create a new file that will contain the information, the file will have 
    #the same name as the original picture 
    #if the file exists then remove it -- sorry
    if os.path.isfile(path+'\\'+file[:-4]+'.txt'):
        os.remove(path+'\\'+file[:-4]+'.txt')           
        
    #now start writing in it
    with open(path+'\\'+file[:-4]+'.txt','a') as f:
        #start writing content in the file 
        #(\n indicates new line), (# indicates that the line will be ignored)
        f.write('# File name { \n')
        f.write(file)
        f.write('\n# } \n')
        
        f.write('# 68 Landmarks [x,y] { \n')
        with open(path+'\\'+'temp_shape.txt','r') as temp_f:
            f.write(temp_f.read())
        f.write('# } \n')
            
        f.write('# Left eye [x,y,r] { \n')
        with open(path+'\\'+'temp_circle_left.txt','r') as temp_f:
            f.write(temp_f.read())
        f.write('# } \n')
            
        f.write('# Right eye [x,y,r] { \n')
        with open(path+'\\'+'temp_circle_right.txt','r') as temp_f:
            f.write(temp_f.read())
        f.write('# } \n')
        
    
    os.remove(path+'\\'+'temp_shape.txt')
    os.remove(path+'\\'+'temp_circle_left.txt')
    os.remove(path+'\\'+'temp_circle_right.txt')
    
#    #we now create a new file, that is formated to be used in the machine 
#    #learning algorithm 
#    if os.path.isfile(path+'\\'+file[:-4]+'_formated.txt'):
#        os.remove(path+'\\'+file[:-4]+'_formated.txt')
#        
#    #now start writing in it
#    with open(path+'\\'+file[:-4]+'_formated.txt','a') as f:
#        f.write('  <image file=\''+file+'\'>\n')
#        top=mod_rect.top()
#        left=mod_rect.left()
#        width=(mod_rect.right()-left)
#        height=(mod_rect.bottom()-top)
#        f.write('    <box top=\''+str(top)+'\' left=\''+str(left)+'\' width=\''+str(width)+'\' height=\''+str(height)+'\'>\n')
#        aux=0
#        for (x,y) in shape:
#            if aux<10:
#                f.write('      <part name=\'0'+str(aux)+'\' x=\''+str(x)+'\' y=\''+str(y)+'\'/>\n')
#            else:
#                f.write('      <part name=\''+str(aux)+'\' x=\''+str(x)+'\' y=\''+str(y)+'\'/>\n') 
#            aux += 1
#            
#        f.write('    </box>\n')
#        f.write('  </image>\n')


    
    