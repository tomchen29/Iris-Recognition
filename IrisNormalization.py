import pickle
import numpy as np

def iris_normalization(data=None, pupils=None, iriss=None):
    def normalization(img, pupil, iris):   
        M = 64
        N = 512
        band = np.zeros((M, N))  
        pupil_row = pupil[0]
        pupil_col = pupil[1]
        pupil_radius = pupil[2]
        iris_row = iris[0]
        iris_col = iris[1]
        iris_radius = iris[2]

        for Y in range(M):
            for X in range(N):

                # calculate theta
                theta = 2*np.pi*X/N

                # get the inner boundary coordinate
                yp = pupil_row + pupil_radius*np.sin(theta)
                xp = pupil_col + pupil_radius*np.cos(theta)

                # get the outer boundary coordinate
                yi = iris_row + iris_radius*np.sin(theta)
                xi = iris_col + iris_radius*np.cos(theta)

                x = min(int(xp + (xi-xp)*Y/M),319)
                y = min(int(yp + (yi-yp)*Y/M),279)      

                band[Y][X] = img[y][x]
        return band
    
    bands = []
    #for i in range(0,len(pupils),1):
    for i in range(0,len(pupils),1):
        img = data.loc[i].values.reshape(280, 320)        
        pupil = pupils[i]
        iris = iriss[i]

        band = normalization(img, pupil, iris)           
        bands.append(band)
        
        if (i+1) % 50 == 0 or i == (len(pupils)-1):
            print("{0} images have been transformed".format(i+1))
                        
    return bands   

def run_normalization_and_save(train_pupils, train_iriss, test_pupils, test_iriss):
    train_normalized = iris_normalization(data=train, pupils=train_pupils, iriss=train_iriss)
    with open('train_normalized', 'wb') as fp:
        pickle.dump(train_normalized, fp)

    test_normalized = iris_normalization(data=test, pupils=test_pupils, iriss=test_iriss)
    with open('test_normalized', 'wb') as fp:
        pickle.dump(test_normalized, fp)   
    
    return train_normalized, test_normalized