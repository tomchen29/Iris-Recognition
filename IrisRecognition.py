import os
import glob 
import cv2
import pandas as pd
import pickle
import numpy as np
from matplotlib import pyplot as plt
from sklearn import metrics
from IrisLocalization import *
from IrisNormalization import *
from ImageEnhancement import *
from FeatureExtraction import *
from IrisMatching import *
from PerformanceEvaluation import *

''' Below is the implementation of data preprocessing, iris localization and iris normalization.'''
''' To save the runtime, I've saved the normalization results to disk. '''
''' You can uncomment and test the code below, which should provide the same or similar result.'''

'''
#### Data Preparation

train_db = []
test_db = []
train_location = []
test_location = []

database_dir = os.path.join(os.getcwd(), 'CASIA Iris Image Database (version 1.0)')
for i in range(1,109,1):
    person_index = '{:03}'.format(i)
    person_imgs = os.path.join(database_dir, person_index)
    train_dir = os.path.join(person_imgs, "1")
    test_dir = os.path.join(person_imgs, "2")

    train_list = glob.glob(os.path.join(train_dir, "*.bmp"))
    test_list =  glob.glob(os.path.join(test_dir, "*.bmp"))
    
    train_location += train_list
    test_location += test_list

    for img_train in train_list:
        img_flat = cv2.imread(img_train, 0).ravel()  
        train_db.append(img_flat)
        
    for img_test in test_list:
        img_flat = cv2.imread(img_test, 0).ravel()  
        test_db.append(img_flat)


train = pd.DataFrame(train_db, columns=[i for i in range(280*320)]) # the image dimension is 280*320
test = pd.DataFrame(test_db, columns=[i for i in range(280*320)]) # the image dimension is 280*320
print("the train and test dataframe have been prepared!")
print("")

f = open("train_location.txt","w+")
for row in train_location:
    f.write(row+"\n") 
f.close()

f = open("test_location.txt","w+")
for row in test_location:
    f.write(row+"\n") 
f.close()

#### Iris Localization
train_pupils, train_iriss = get_pupil_and_iris_circle(mode='train',dataset=train)
test_pupils, test_iriss = get_pupil_and_iris_circle(mode='test',dataset=test)
print("the train and test pupil/iris have been localized completely!")
print("")

#### Iris Normalization (original code)
train_normalized, test_normalized = run_normalization_and_save(train_pupils, train_iriss, test_pupils, test_iriss)
print("the normalization has been completely!")
print("")

'''

#### Iris Normalization (load from files)
with open('train_normalized', 'rb') as fp:
    train_normalized = pickle.load(fp)
    
with open('test_normalized', 'rb') as fp:
    test_normalized = pickle.load(fp)

print("Successfully loaded the normalized results.")
print("")

#### Image Enhancement
enhanced_list_train = []
enhanced_list_test = []
for img in train_normalized:
    enhanced_list_train.append(iris_enhancement(img, illumination=False))

for img in test_normalized:
    enhanced_list_test.append(iris_enhancement(img, illumination=False))
feature_extracted_train = execute_feature_extraction(enhanced_list_train)
feature_extracted_test = execute_feature_extraction(enhanced_list_test)
print("Image Enhancement completed.")
print("")


#### Iris Matching
print("Preparing Iris Matching data...")
print("")
train_X = np.array(feature_extracted_train)
train_Y = np.array([(i//3+1) for i in range(train_X.shape[0])])
test_X = np.array(feature_extracted_test)
test_Y = np.array([(i//4+1) for i in range(test_X.shape[0])])
print("Iris Matching data have been completed.")
print("")

## Prediction Method 0: Original Feature Vector, as denoted in the paper
values_Org, predictions_Org = nearest_centroid_classifier(train_X, test_X, lda=None)
print("prediction using the original feature vector has been completed!")
print("")

## Prediction Method I: Simple LDA, as denoted in the paper
lda_org = LDA().fit(train_X, train_Y)   
values_LDA, predictions_LDA = nearest_centroid_classifier(train_X, test_X, lda=lda_org)
print("prediction using the LDA model has been completed!")
print("")

## Prediction Method II: Standardization+ PCA + LDA
print("preparing for the improved PCA+LDA model...")

values_PCA, predictions_PCA = PCA_LDA_model(train_X, train_Y, test_X, test_Y)
print("prediction using the improved PCA+LDA model has been completed!")
print("")

#### Evaluation Part I: Iris Identification Evaluation
## RRC Table
print("preparing for the RRC table...")
print("")
originals = evaluate(predictions_Org, test_Y)
transforms1 = evaluate(predictions_LDA, test_Y)
transforms2 = evaluate(predictions_PCA, test_Y)
plot_CRR(originals, transforms1, transforms2)
print("")

## RRC using features of different dimensionality
print("preparing for the RRC tuning curve...")
print("")
n_arr = [i for i in range(10, 108, 10)]+[107]
recognition_rates = np.empty(len(n_arr))
for i in range(len(n_arr)):
    lda = LDA(n_components=n_arr[i])
    lda.fit(train_X, train_Y) 
    max_recognition_rate = np.amax(evaluate(nearest_centroid_classifier(train_X, test_X, lda=lda)[1], test_Y))
    recognition_rates[i] = max_recognition_rate
plot_LDA_tunning(n_arr, recognition_rates)

#### Evaluation Part II: Iris Verification Evaluation

## FMR-FNMR table
print("preparing for the FMR-FNMR table...")
print("")
cos_dist = np.asarray(values_LDA)[:,2]
cos_prediction = np.asarray(predictions_LDA)[:,2]

print("preparing for the point estimators...")
print("")
fm1, fnm1, tpr1, fpr1 = metrics_calculator(cos_dist, cos_prediction, 0.446)
fm2, fnm2, tpr2, fpr2 = metrics_calculator(cos_dist, cos_prediction, 0.472)
fm3, fnm3, tpr3, fpr3 = metrics_calculator(cos_dist, cos_prediction, 0.502)

print("preparing for the confience interval estimators...")
print("")
(boostrap_fms, boostrap_fnms, _, _) = boostrap(cos_dist, cos_prediction, 0.446)
boostrap_fms_446_lb, boostrap_fms_446_up = confidence_interval_boostrap(boostrap_fms)
boostrap_fnms_446_lb, boostrap_fnms_446_up = confidence_interval_boostrap(boostrap_fnms)

(boostrap_fms, boostrap_fnms, _, _) = boostrap(cos_dist, cos_prediction, 0.472)
boostrap_fms_472_lb, boostrap_fms_472_up = confidence_interval_boostrap(boostrap_fms)
boostrap_fnms_472_lb, boostrap_fnms_472_up = confidence_interval_boostrap(boostrap_fnms)

(boostrap_fms, boostrap_fnms, _, _) = boostrap(cos_dist, cos_prediction, 0.502)
boostrap_fms_502_lb, boostrap_fms_502_up = confidence_interval_boostrap(boostrap_fms)
boostrap_fnms_502_lb, boostrap_fnms_502_up = confidence_interval_boostrap(boostrap_fnms)

print("preparing for the final FMR-FNMR table...")
print("")
table = PrettyTable()
table.field_names = ["Threshold", "False match rate (%)", "False non-match rate (%)"]
table.add_row([0.446, "{0:.3f} [{1:.3f},{2:.3f}]".format(fm1, boostrap_fms_446_lb,boostrap_fms_446_up), 
               "{0:.3f} [{1:.3f},{2:.3f}]".format(fnm1, boostrap_fnms_446_lb,boostrap_fnms_446_up)])

table.add_row([0.472, "{0:.3f} [{1:.3f},{2:.3f}]".format(fm2, boostrap_fms_472_lb,boostrap_fms_472_up), 
               "{0:.3f} [{1:.3f},{2:.3f}]".format(fnm2, boostrap_fnms_472_lb,boostrap_fnms_472_up)])

table.add_row([0.502,"{0:.3f} [{1:.3f},{2:.3f}]".format(fm3, boostrap_fms_502_lb,boostrap_fms_502_up), 
               "{0:.3f} [{1:.3f},{2:.3f}]".format(fnm3, boostrap_fnms_502_lb,boostrap_fnms_502_up)])
print("False Match and False Nonmatch Rates with Different Threshold Values")
print(table)
print("")

## Curve Evaluation (cosine distance)
print("Recording the false_match_rate, false_non_match_rate, true_positive_rate, false_positive_rate with different threadsholds...")
print("")
threadshold_arr = [i for i in np.arange(0.12, 0.78, 0.02)]
fms = []; fnms = []; tprs = []; fprs = []
for threadshold_i in threadshold_arr:
    fm, fnm, tpr, fpr = metrics_calculator(cos_dist, cos_prediction, threadshold_i)
    fms.append(fm)
    fnms.append(fnm)
    tprs.append(tpr)
    fprs.append(fpr)

# ROC Curve
print("preparing the ROC curve...")
print("")
plot_curve(fprs, tprs, 
         'ROC curve (area = %0.2f)' % metrics.auc(fprs, tprs, reorder=False), 
         "False Positive Rate", "True Positive Rate", "ROC Curve")

# FMR and FNMR Curve
print("preparing the FMR-FNMR curve...")
print("")
plot_curve(fms, fnms,  None, 
           "False Match Rate (%)", "False Non-Match Rate (%)", "FMR and FNMR Curve",'r')







