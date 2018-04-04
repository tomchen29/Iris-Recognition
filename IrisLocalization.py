import numpy as np
import os
from scipy.io import loadmat

## use a historgram method to estimate the area of pupil
def binarize_by_pupil(img):
    threshold = sorted(img)[4000]
    def binarize(img):
        new_img = 0 if img <= threshold else 255
        return new_img
    binarize = np.vectorize(binarize)
    return binarize(img)


## use the 'pure-black' data ponts' locations after binarization to estimate the center
def find_pupil_center(img_vector):
    counter = 0
    row_total = 0.0
    col_total = 0.0
    for i in range(len(img_vector)):
        if img_vector[i] == 0:    # if the point lies in our estimated pupil zone
            row_i = int(i/320)    # then get the point's coordinate
            col_i = i % 320
            
            row_total += row_i
            col_total += col_i
            counter+=1

    row_index = int(row_total/counter)  # calculate the average row index
    col_index = int(col_total/counter)  # calculate the average coloum index
    return (row_index, col_index)

## find the optimal pupil and iris circle
def get_pupil_and_iris_circle(mode=None, dataset=None):
    MatLab_Circle_Train = os.path.join(os.getcwd(), 'MatLab_Circle_'+mode)
    counter = 0
    pupils = []
    iriss = []
    
    # get all the estimated circles acqured by Matlab
    for i in range(1, dataset.shape[0]+1, 1):    
        fname = os.path.join(MatLab_Circle_Train, mode+'_circle_'+str(i)+'.mat')
        result = loadmat(fname)['c'][0]
        
        pupil_centers = result[0]
        pupil_radiis = result[1]
        
        if len(result) == 5:
            iris_centers = result[2]
            iris_radiis = result[3]  
            img_name = result[4]
        else:  # Bad image quality. Have to estimate the iris circle based on pupil center
            print("the "+str(i)+" image does not find iris. Need to estimate its value!")
            iris_centers = []
            iris_radiis = []
            for j in range(len(pupil_centers)):
                iris_radiis.append([2*pupil_radiis[j][0]])
                iris_centers.append([pupil_centers[j][0], pupil_centers[j][1]])                             
        
        
        #### get the corresponding image data
        img = dataset.loc[i-1].values

        #### get the best iris center and radius
        #### We use the min distance to estimated pupil center to approximately estimate the best iris circle
        pupil_estimated_center = find_pupil_center(binarize_by_pupil(img.ravel()))    
        
        best_iris_x = None
        best_iris_y = None
        best_iris_radius = None
        min_dist = 1000
        for j in range(len(iris_centers)):
            iris_x = iris_centers[j][1]
            iris_y = iris_centers[j][0]
            iris_radius = iris_radiis[j][0]  
            dist = np.sqrt((iris_x-pupil_estimated_center[0]) ** 2 + (iris_y -pupil_estimated_center[1]) ** 2)
            if dist < min_dist:
                min_dist = dist
                best_iris_x = iris_x
                best_iris_y = iris_y
                best_iris_radius = iris_radius
                
            
        #### get the best pupil center and radius, based on the min distance to the estimated center
        # intialize the data
        best_pupil_x = None
        best_pupil_y = None
        best_pupil_radius = None
        min_dist = 300  # threadshold. if the distance is larger than this, then there is probably a problem
        
        # search for the best pupil circle
        search_counter = 0
        while best_pupil_x is None:
            if search_counter > 0:  # if we do a bad segmentation and can't find pupil inside the iris
                best_iris_radius+=5 # manually enlarge the iris circle       
            for j in range(len(pupil_centers)):
                pupil_x = pupil_centers[j][1]
                pupil_y = pupil_centers[j][0]
                pupil_radius = pupil_radiis[j][0]

                if ((pupil_x+pupil_radius <= best_iris_x+best_iris_radius) and (pupil_x-pupil_radius >= best_iris_x-best_iris_radius) 
                    and (pupil_y+pupil_radius <= best_iris_y+best_iris_radius) and (pupil_y-pupil_radius >= best_iris_y-best_iris_radius)):
                    dist = np.sqrt((pupil_x-pupil_estimated_center[0]) ** 2 + (pupil_y -pupil_estimated_center[1]) ** 2)
                    if dist < min_dist:
                        min_dist = dist
                        best_pupil_x = pupil_x
                        best_pupil_y = pupil_y
                        best_pupil_radius = pupil_radius
            search_counter+=1

        ## append the results
        if best_iris_x is not None and best_iris_y is not None and best_iris_radius is not None:
            iris = [best_iris_x, best_iris_y, best_iris_radius]
            iriss.append(iris)
        else:
            print(fname)
            raise ValueError('iris has one requried value missing. Check it!!')       
            
        if best_pupil_x is not None and best_pupil_y is not None and best_pupil_radius is not None:
                pupil = [best_pupil_x, best_pupil_y, best_pupil_radius]
                pupils.append(pupil)       
        else:       
            print(fname)
            raise ValueError('pupil has one requried value missing. Check it!!')
            
    
    return (pupils, iriss)