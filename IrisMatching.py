import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.neighbors.nearest_centroid import NearestCentroid
from skimage.measure import block_reduce
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings("ignore")

def nearest_centroid_classifier(train_X, test_X, lda = None):   
    
    ## get the L1, L2, COS distance between f and fis
    def get_distance_matrix(f, fis):
        values_mat = np.zeros((fis.shape[0],3))

        for i in range(fis.shape[0]):
            fi = fis[i].reshape(fis.shape[1],1)
            d1 = abs(f-fi).sum()                     
            d2 = np.asscalar( (f-fi).T.dot(f-fi) )
            d3 = 1 - np.asscalar(f.T.dot(fi)) / (np.linalg.norm(f)*np.linalg.norm(fi))
            values_mat[i,:] = np.array([d1,d2,d3])   
        return values_mat
    
    test_target = test_X
    fis = block_reduce(train_X, (3,1), np.mean)  # use mean to calculate the fi
    
    ## if we want to apply LDA for dimension reduction
    if lda is not None:
        test_target = lda.transform(test_target) 
        fis = lda.transform(fis)
 
    ## get the prediction and value list
    prediction_list = []
    values_list = []
    #distance_list = []
    
    ## loop to get each f in the test target
    for i in range(test_target.shape[0]):
        
        ## calculate the three distances between f and fi, then predict f's class by the min dist index
        f = test_target[i,:].reshape(test_target.shape[1],1)
        dist_mat = get_distance_matrix(f,fis)   # the distances matrix of f and fis
        min_index = np.argmin(dist_mat,axis=0)  # the min idnex of the distances marix
        prediction = min_index + 1              # the predictions of f
        
        ## get the value of prediction
        values = []
        value_matrix = dist_mat[min_index]
        values.append(value_matrix[0][0])
        values.append(value_matrix[1][1])
        values.append(value_matrix[2][2])
        
        ## append the results to the output lists
        prediction_list.append(prediction)
        values_list.append(values)
        #distance_list.append(values_mat)
    
    return (values_list, prediction_list)

def evaluate(prediction_list, Y):
    ## create a 1x3 list to save the total number of the 
    # correct classification of the three distances
    true_classificaton = np.zeros(3)  
    for i in range(len(Y)):
        prediction = prediction_list[i]
        if prediction[0] == Y[i]:
            true_classificaton[0]+=1
        if prediction[1] == Y[i]:
            true_classificaton[1]+=1
        if prediction[2] == Y[i]:
            true_classificaton[2]+=1
    return true_classificaton/Y.shape[0]

def PCA_LDA_model(train_X, train_Y, test_X, test_Y):
    
    ##### First, standardize train and test data to gain better result
    train_X_std = StandardScaler().fit_transform(train_X)
    test_X_std = StandardScaler().fit_transform(test_X)
    
    ##### Second,tune the best n component for PCA
    n_arr = [i for i in range(10, 324, 10)]  # number of componenets to tune
    recognition_rates = np.empty(len(n_arr)) # compare list of the recognition_rate with i components
    
    for i in range(len(n_arr)):
        ## use the tuned PCA model to perform the first-time feature dimension
        pca = PCA(n_components=n_arr[i]).fit(train_X_std)
        train_X_tf_std = pca.transform(train_X_std)
        test_X_tf_std = pca.transform(test_X_std)

        ## use the reduced train dataset to fit a lda model for the second-time feature dimention
        lda=LDA().fit(train_X_tf_std, train_Y)

        ## get the prediction_list and save it into the compare list
        prediction_list = nearest_centroid_classifier(train_X_tf_std, test_X_tf_std, lda)[1]
        max_recognition_rate = np.amax(evaluate(prediction_list, test_Y))
        recognition_rates[i] = max_recognition_rate
    
    ##### Third, perform the optimal PCA transformation on data and refit the LDA model
    best_components = n_arr[np.argmax(recognition_rates)]
    print("the optimal number of component is: ", best_components)
    pca = PCA(n_components=best_components).fit(train_X_std)
    train_X_tf_std = pca.transform(train_X_std)
    test_X_tf_std = pca.transform(test_X_std)

    ##### Finally, retrain the nearest centroid classifier with the refitted LDA model
    lda=LDA().fit(train_X_tf_std, train_Y)
    
    #### get the output
    values_PCA, predictions_PCA = nearest_centroid_classifier(train_X_tf_std, test_X_tf_std, lda)
    
    return values_PCA, predictions_PCA