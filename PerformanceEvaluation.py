from prettytable import PrettyTable
from matplotlib import pyplot as plt
from random import randint
import numpy as np

def plot_CRR(originals, transforms1, transforms2):
    ## convert the input data to 100% scale
    originals = originals*100
    transforms1 = transforms1*100
    transforms2 = transforms2*100
    
    ## construct the table
    table = PrettyTable()
    table.field_names = ["Similarity Measurement", "Original feature Set", "Reduced feature set (only LDA)", "Reduced feature set (PCA+LDA)"]
    table.add_row(["L1 distance measure",originals[0],transforms1[0], transforms2[0]])
    table.add_row(["L2 distance measure", originals[1], transforms1[1], transforms2[1]])
    table.add_row(["Cosine similarity measure", originals[2], transforms1[2], transforms2[2]])
    print("TABLE of Correct Recognition Rate (%) Using Different Similarity Measures")
    print(table)

def plot_LDA_tunning(tunning_values, rates):
    fg = plt.figure(figsize=(5, 5))
    ax = fg.add_subplot(111)

    ax.set_xlabel("Dimensionality of the feature vector", size="large")
    ax.set_ylabel("Correct regonition rate", size="large")

    line_cv = ax.plot(tunning_values, rates, label="recognition rate", marker='o')

    ax.legend(loc="best", fontsize="large")
    plt.show()

def metrics_calculator(cos_dist, cos_prediction, threashold, boostrap=False):  
    TP = 0; FP = 0; TN = 0; FN = 0
    false_match_rate = None
    false_non_match_rate = None   
    true_positive_rate = None
    false_positive_rate = None
    total = len(cos_dist)
    divider = 4 if boostrap==False else 1
    
    ## calculate the TP, FP, TN, FN values 
    for i in range(total):                
        if cos_dist[i] < threashold:   #match
            if cos_prediction[i] == (i//divider)+1:  #correct prediction
                TP += 1
            else:  #wrong prediction
                FP += 1

        else:   #non match
            if cos_prediction[i] == (i//divider)+1:  #correct prediction           
                FN += 1
            else:  #wrong prediction
                TN += 1
    
    ## calculate the false_match_rate, false_non_match_rate, true_positive_rate, false_positive_rate
    if TP > 0 or FP > 0:
        false_match_rate = FP/(TP+FP)
    if TN > 0 or FN > 0:
        false_non_match_rate = FN/(TN+FN)   
    if TP > 0 or FN > 0:
        true_positive_rate = TP/(TP+FN)
    if FP > 0 or TN > 0:
        false_positive_rate = FP/(FP+TN)
    
    return (false_match_rate, false_non_match_rate, true_positive_rate, false_positive_rate)

def boostrap(cos_dist, cos_prediction, threashold):
    boostrap_fms = []
    boostrap_fnms = []
    boostrap_tprs = []
    boostrap_fprs = []

    for k in range(5000):  ## Boostrap 5000 times
        selected_case = np.array([randint(0,3) for i in range(108)])
        intervals = np.array([i for i in range(0, 432, 4)])
        selected_index = selected_case+intervals
        
        new_cos_dist = cos_dist[selected_index]
        new_cos_prediction = cos_prediction[selected_index]
        
        false_match_rate, false_non_match_rate, true_positive_rate, false_positive_rate = metrics_calculator(
            new_cos_dist, new_cos_prediction, threashold, boostrap=True)
        
        boostrap_fms.append(false_match_rate)
        boostrap_fnms.append(false_non_match_rate)
        boostrap_tprs.append(true_positive_rate)
        boostrap_fprs.append(false_positive_rate)
        
    return (boostrap_fms, boostrap_fnms, boostrap_tprs, boostrap_fprs)

def confidence_interval_boostrap(boostrap_result, alpha=0.05):
    asc_list = sorted(boostrap_result)
    lower_bound_index = int(len(asc_list)*alpha/2)
    upper_bound_index = int(len(asc_list)*(1-alpha/2))
    lb = asc_list[lower_bound_index]
    up = asc_list[upper_bound_index]
    return (lb, up)

def plot_curve(fprs, tprs, label, xlabel, ylabel, title, color='b'):
    plt.plot(fprs,tprs, label=label, color=color)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title, fontsize=20)
    plt.legend(loc='best')
    plt.show()