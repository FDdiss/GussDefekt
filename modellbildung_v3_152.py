from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectFromModel

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from prepare_datasets_v2 import prepare_datasets



from sklearn.utils import resample

from datetime import datetime
import xgboost as xgb
from xgboost import DMatrix
#from xgboost import train
from xgboost import plot_importance
import shap

import csv
import math
import random
import pickle
import os
import numpy as np
import matplotlib.pyplot as pl
import pandas as pd
import joblib
import time
import sys
import os
from itertools import product


#list1=[1,2,3,4]
#list2=["a","b","c","d"]
#list3=[".",",","-","_"]
#for i,j,k in product(list1,list2,list3):
#    print(i,j,k)

path="E:\dob\GussfehlerKI"
#this_dir = os.path.dirname(__file__) # Path to loader.py
#sys.path.append(os.path.join(this_dir, path+"/all_Features_20190215"))
#### run02: erster durchlauf für videopräsi
### run 03: voschlag: gridsearch von colsample tree/node/level (lower) subsample (lower), max depth (lower) gamma (higher) eta (lower) minchildweight (larger). early stopping drastisch reduziert auf 10

runname="run04"
#folderexplanation="NIO_default"
#folderexplanation="NIO_default_hypopt"
#folderexplanation="NIO_hypopt_fe"
#folderexplanation="NIO_hypopt_tsfresh"
#folderexplanation="NIO_hypopt_XGBfiltered"

#folderexplanation="Defects_hypopt"
#folderexplanation="Defects_hypopt_fe"
#folderexplanation="Defects_hypopt_tsfresh"
folderexplanation="baseline_NIOIO"

#folderexplanation="Defects_hypopt_koknest"
#folderexplanation="Defects_hypopt_fe_koknest"
#folderexplanation="Defects_hypopt_tsfresh_koknest"
#folderexplanation="Defects_hypopt_XGBfiltered_koknest"
#folderexplanation="test"

continue_trigger=False

best_export_trigger=True
XGB_modelsave_trigger = False
SHAP_trigger = False
loss_curve_trigger=False
importancesave_trigger= False


NN_trigger = True
XGB_trigger = True
xgbfilter_trigger=True
xgbfraction_list=[100,80,50,25,10,5,2,1]   ### in %

balancing_trigger = True
validation_trigger=True
validation_fraction=0.1


base_trigger = True
base_fe_trigger = True
base_tsfresh_trigger = True

kok_base_trigger = True
kok_fe_trigger = True
kok_tsfresh_trigger = True

nest_base_trigger = True
nest_fe_trigger = True
nest_tsfresh_trigger = True



targeterrors=["Gasblase","undicht","Poros",["Gasblase","Kaltlauf","undicht","Poros"],"Kaltlauf"] #targeterrors=["Gasblase","Kaltlauf","Kernfehler","Lunker","undicht","Poros","Prozessfehler","Sonstiges"]
targeterrors=targeterrors[3:4]




scalerlist=["",preprocessing.RobustScaler,preprocessing.StandardScaler,preprocessing.MinMaxScaler]
scalerlist=[scalerlist[1]]
scalernamelist=["none","robust","standard","minmax"]
scalernamelist=scalernamelist[1:2]

randomseed=3

### XGBoost Parameter

XGB_objective=["binary:logistic"] #O1:log (objective, eta:0.1783)  #XGB_objective=["reg:logistic","reg:squarederror","reg:squaredlogerror","reg:pseudohubererror"] #O1:log (objective, eta:0.1783)
XGB_seed=[randomseed]
XGB_verbosity=[0]

XGB_eta=[0.3] #O1:0.1 (objective, eta: 0.1783) #XGB_eta=[0.05,0.1,0.2,0.3] #O1:0.1 (objective, eta: 0.1783)
XGB_alpha=[0.0005]  #XGB_alpha=[0,0.25,0.5,1.0,2.0]
XGB_gamma=[0.0005]  #XGB_gamma=[0.01,0.05,0.1,0.2,0.3,1.0,2.0,5.0,10]
XGB_lambda=[1]  #XGB_lambda=[1,2,4,8]
XGB_minchild_weight=[1]   #XGB_minchild_weight=[0.01,0.05,0.1,0.2,0.3,1.0,2.0,5.0,10]
XGB_max_delta_step=[0]  #XGB_max_delta_step=[0.5,1,2,4,8,16]

XGB_n_estimators=[5000] #500
XGB_earlystoppingrounds=[500]
XGB_parallel_tree=[1]  #XGB_parallel_tree=[2,4,6,8,10,50]
XGB_maxdepth=[4] #XGB_maxdepth=[2,3,4,5,6,7,8,9,10,16,32]

XGB_subsample=[0.75]   #XGB_subsample=[0.25,0.5,0.75,1.0]
XGB_colsample_bytree=[0.75]    #XGB_colsample_bytree=[0.25,0.5,0.75,1.0]
XGB_colsample_bylevel=[0.75]   #XGB_colsample_bylevel=[0.25,0.5,0.75,1.0]
XGB_colsample_bynode=[0.75]    #XGB_colsample_bynode=[0.25,0.5,0.75,1.0]
# for XGB_objective_i,XGB_seed_i,XGB_eta_i,XGB_alpha_i,XGB_gamma_i,XGB_lambda_i,XGB_minchild_weight_i,XGB_max_delta_step_i,XGB_n_estimators_i,XGB_earlystoppingrounds_i,XGB_parallel_tree_i,XGB_maxdepth_i,XGB_subsample_i,XGB_colsample_bytree_i,XGB_colsample_bylevel_i,XGB_colsample_bynode_i in XGB_objective,XGB_seed,XGB_eta,XGB_alpha,XGB_gamma,XGB_lambda,XGB_minchild_weight,XGB_max_delta_step,XGB_n_estimators,XGB_earlystoppingrounds,XGB_parallel_tree,XGB_maxdepth,XGB_subsample,XGB_colsample_bytree,XGB_colsample_bylevel,XGB_colsample_bynode
#NN Parameter

NN_verbose=[True]

#NN_layers=[(10,10),(20,20),(30,30),(40,40),(50,50),(100,100),(200,200),(400,400),(800,800),
#           (10,5),(20,10),(30,15),(40,20),(50,25),(100,50),(200,100),(400,200),(800,400),
#           (10,10,5),(20,20,10),(30,30,15),(40,40,20),(50,50,25),(100,100,50),(200,200,100),(400,400,200),(800,800,400)
#          ]
#NN_layers_names=[
#    "10x10","20x20","30x30","40x40","50x50","100x100","200x200","400x400","800x800",
#    "10x5","20x10","30x15","40x20","50x25","100x50","200x100","400x200","800x400",
#    "10x10x5","20x20x10","30x30x15","40x40x20","50x50x25","100x100x50","200x200x100","400x400x200","800x800x400"
#    ]
NN_layers=[(100,100)]    
NN_layers_names=["100x100"]
NN_activation=["relu"] #default='relu'
NN_solver=["adam"] #default='adam'
NN_alpha=[0.0005] # default=0.0001
#NN_batch_size=["auto"] # default='auto' # Ersetzt durch eine primzahlenberechnung, um 
NN_minibatch_size = [32] # integer, not a list! wird zur Berechnung der genauen minibatchgröße benutzt, um eine möglichst ideale Teilung zu ermöglichen
NN_learning_rate=["constant"] #default='constant'
NN_learning_rate_init=[0.0001] #default=0.001 ### 0.00001,0.001

NN_early_stopping=[True]
NN_tol=[0.00001] #default 1e-4
NN_n_iter_no_change = [500] #default 10 ###

NN_validation_fraction=[0.1] #default 0.1
NN_max_fun=[15000] # default=15000
NN_max_iter=[5000]#default =200
NN_beta_1=[0.9]#default =0.9, 
NN_beta_2=[0.999]#default =0.999, 
NN_epsilon=[1e-08]#default =1e-08, 

NN_momentum=[0.9]
NN_nesterovs_momentum=[True]
NN_random_state=[randomseed]
NN_power_t=[0.5] #default 0.5
NN_shuffle=[True] #default True


##############################################################################################
# Start MAIN
##############################################################################################
random.seed(randomseed)
triggerlist=[base_trigger,base_fe_trigger,base_tsfresh_trigger,kok_base_trigger,kok_fe_trigger,kok_tsfresh_trigger,nest_base_trigger,nest_fe_trigger,nest_tsfresh_trigger]
if xgbfilter_trigger==False:
    xgbfraction_list=[100]
globalscores=[]
### Ordner erstellen
if folderexplanation != "":
    savepath=path+"/"+runname+"_"+folderexplanation
else:
    savepath=path+"/"+runname
if not os.path.exists(savepath):
      
    # if the demo_folder directory is not present 
    # then create it.
    os.makedirs(savepath)
elif continue_trigger == True:
    lastcsv="none"
    scorefile_searchstring = "scores"

    for filename in os.listdir(savepath):
        if scorefile_searchstring in filename:
            lastcsv=str(savepath + os.sep +filename)
    if lastcsv != "none":
        with open(lastcsv, 'r') as fh:
            csvdata=pd.read_csv(fh)
        csvdata.drop(csvdata.columns[0],axis=1,inplace=True)
        globalscores = csvdata.values.tolist()
        print(globalscores)

### Daten laden

with open(path+"/totaldata_13022023_newnames.csv") as fh:
    orgdata=pd.read_csv(fh)
    orgdata.drop(orgdata.columns[0],axis=1,inplace=True)

### preprocessing

#Spalten mit DMC löschen 
orgdata.drop("DMCODE",axis=1,inplace=True)
#thermocorellation hat ein paar leerwerte. Werden mit 0 aufgefüllt
orgdata=orgdata.fillna(0)

###### Primzahlen finden
primelist=[2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97,101,103,107,109,113,127,131,137,139,149,151,157,163,167,173,179,181,191,193,197,199,211,223,
           227,229,233,239,241,251,257,263,269,271,277,281,283,293]


columnnames=[
        "model","runname","dataname","xgbfraction","errorname","balancing","scalername","randomseeds",
        
        "count_features","count_rows_all",
        "count_rows_balanced","count_rows_balanced_IO","count_rows_balanced_NIO",
        
        "count_IO_test","count_NIO_test",
        "balanced_acc_test","precision_NIO_test","recall_NIO_test","fscore_NIO_test","precision_IO_test","recall_IO_test","fscore_IO_test","acc_test",
        "count_IO_test_best","count_NIO_test_best",
        "balanced_acc_test_best","precision_NIO_test_best","recall_NIO_test_best","fscore_NIO_test_best","precision_IO_test_best","recall_IO_test_best","fscore_IO_test_best","acc_test_best",                           

        "count_IO_all","count_NIO_all",
        "balanced_acc_all","precision_NIO_all","recall_NIO_all","fscore_NIO_all","precision_IO_all","recall_IO_all","fscore_IO_all","acc_all",
        "balanced_acc_all_best","precision_NIO_all_best","recall_NIO_all_best","fscore_NIO_all_best","precision_IO_all_best","recall_IO_all_best","fscore_IO_all_best","acc_all_best",    

        "count_IO_val","count_NIO_val",
        "balanced_acc_val","precision_NIO_val","recall_NIO_val","fscore_NIO_val","precision_IO_val","recall_IO_val","fscore_IO_val","acc_val",
        "balanced_acc_val_best","precision_NIO_val_best","recall_NIO_val_best","fscore_NIO_val_best","precision_IO_val_best","recall_IO_val_best","fscore_IO_val_best","acc_val_best",

        "count_IO_true_test_data","count_IO_false_test_data","count_NIO_false_test_data","count_NIO_true_test_data",
        "count_IO_true_test_data_best","count_IO_false_test_data_best","count_NIO_false_test_data_best","count_NIO_true_test_data_best", 
        "count_IO_true_val_data","count_IO_false_val_data","count_NIO_false_val_data","count_NIO_true_val_data",
        "count_IO_true_val_data_best","count_IO_false_val_data_best","count_NIO_false_val_data_best","count_NIO_true_val_data_best",
        "count_IO_true_all_data","count_IO_false_all_data","count_NIO_false_all_data","count_NIO_true_all_data",
        "count_IO_true_all_data_best","count_IO_false_all_data_best","count_NIO_false_all_data_best","count_NIO_true_all_data_best",
        "count_IO_train","count_NIO_train","count_IO_train_best","count_NIO_train_best",

        "modeltrainingloss","modelvalidationloss",
        "modeltrainingloss_best","modelvalidationloss_best",     

        "NN_solver","NN_activation","NN_layers","NN_learning_rate_init","NN_alpha","NN_minibatchsize","NN_max_iter","NN_n_iter_no_change","NN_tol",
        "NN_validation_fraction","NN_beta_1","NN_beta_2","NN_epsilon","NN_max_fun","NN_learning_rate","NN_shuffle","NN_early_stopping","NN_power_t","NN_momentum","NN_nesterovs_momentum",
        "XGB_objective","XGB_n_estimators","XGB_parallel_tree","XGB_maxdepth","XGB_eta","XGB_earlystoppingrounds","XGB_minchild_weight","XGB_max_delta_step",
        "XGB_subsample","XGB_colsample_bytree","XGB_colsample_bylevel","XGB_colsample_bynode","XGB_alpha","XGB_lambda","XGB_gamma"
        ]





#                            ["NN"]+[runname]+[dataname]+[errorname]+
#                            [balanced_acc_test]+[acc_test]+[count_NIO_test]+prfs_NIO_test[:3]+[count_IO_test]+prfs_IO_test[:3]+
##                            IONIO_confusion_test+
#                            [balanced_acc_all]+[acc_all]+[count_NIO_all]+prfs_NIO_all[:3]+[count_IO_all]+prfs_IO_all[:3]+
#                            IONIO_confusion_all+
#                            [balancing]+[scalername]+[modeltrainingloss]+[modelvalidationloss]+
#                            [NN_solver[0]]+[NN_activation[0]]+[NN_layers[l]]+[NN_learning_rate_init[0]]+[NN_alpha[n]]+[NN_minibatch_size]+[NN_max_iter[0]]+[NN_n_iter_no_change[0]]+[NN_tol[0]]+
#                            [NN_validation_fraction[0]]+[NN_beta_1[0]]+[NN_beta_2[0]]+[NN_epsilon[0]]+[NN_max_fun[0]]+[NN_learning_rate[0]]+[NN_shuffle[0]]+[NN_random_state[0]]+[NN_early_stopping[0]]+[NN_power_t[0]]+
#                            ["none"]+["none"]+["none"]+["none"]+["none"]+["none"]+["none"]+["none"]+["none"]+ ["none"]+["none"]+["none"]+["none"]+["none"]+["none"]+["none"]

##############################################################################################################################
#start Rechnungen
##############################################################################################################################

random.seed(3)
randomseeds=random.sample(range(1, 1000), 10)
randomseeds_str='|'.join([str(elem) for elem in randomseeds])

saveslotcounter=1
best_dict={}
best_score_dict={}
balanced_acc_val="none"
acc_val="none"
count_IO_true_val_data="none"
count_IO_false_val_data="none"
count_NIO_false_val_data="none"
count_NIO_true_val_data="none"
count_NIO_val="none"
count_IO_val="none"
precision_IO_val="none"
recall_IO_val="none"
fscore_IO_val="none"
precision_NIO_val="none"
recall_NIO_val="none"
fscore_NIO_val="none" 
balanced_acc_val_best="none"
acc_val_best="none"
count_IO_true_val_data_best="none"
count_IO_false_val_data_best="none"
count_NIO_false_val_data_best="none"
count_NIO_true_val_data_best="none"
count_NIO_val_best="none"
count_IO_val_best="none"
precision_IO_val_best="none"
recall_IO_val_best="none"
fscore_IO_val_best="none"
precision_NIO_val_best="none"
recall_NIO_val_best="none"
fscore_NIO_val_best="none"  
 
for targeterror in targeterrors:   
    if type(targeterror)==list:
        errorname="NIO"
    else:
        errorname=targeterror
    XGBfiltermodel_base=0
    XGBfiltermodel_fe=0 
    XGBfiltermodel_tsfresh=0 

    for xgbfraction in xgbfraction_list:
        if ((True in [base_trigger,kok_base_trigger,nest_base_trigger]) and (xgbfraction!=100) and (XGBfiltermodel_base==0)):
            XGBfiltermodel_base_path = savepath+os.sep+"base_data_100_"+errorname+"_XGB_bestmodel"

            if os.path.exists(XGBfiltermodel_base_path):

                XGBfiltermodel_base = joblib.load(XGBfiltermodel_base_path)
        if ((True in [base_fe_trigger,kok_fe_trigger,nest_fe_trigger]) and (xgbfraction!=100) and (XGBfiltermodel_fe==0)):
            XGBfiltermodel_fe_path = savepath+os.sep+"fe_data_100_"+errorname+"_XGB_bestmodel"   
            if os.path.exists(XGBfiltermodel_fe_path):

                XGBfiltermodel_fe = joblib.load(XGBfiltermodel_fe_path)             
        if ((True in [base_tsfresh_trigger,kok_tsfresh_trigger,nest_tsfresh_trigger]) and (xgbfraction!=100) and (XGBfiltermodel_tsfresh==0)):
            XGBfiltermodel_tsfresh_path = savepath+os.sep+"tsfresh_data_100_"+errorname+"_XGB_bestmodel"
            if os.path.exists(XGBfiltermodel_tsfresh_path):

                XGBfiltermodel_tsfresh = joblib.load(XGBfiltermodel_tsfresh_path)    
        datalist,datanamelist,xgbfraction_datalist=prepare_datasets(orgdata,triggerlist,xgbfraction,XGBfiltermodel_base,XGBfiltermodel_fe,XGBfiltermodel_tsfresh)
        for dataname in datanamelist:
            entryname=str(dataname)+"_"+str(xgbfraction).zfill(3)+"_"+errorname+"_NN"
            best_dict[entryname]=0
            entryname=str(dataname)+"_"+str(xgbfraction).zfill(3)+"_"+errorname+"_XGB"
            best_dict[entryname]=0
        for k in range(len(datalist)):
            dataname=datanamelist[k]
            xgbfraction=xgbfraction_datalist[k]
            print("jetzt:",dataname)
            data=datalist[k].copy()

            print("jetzt:",dataname,errorname,xgbfraction)
            #select all rows with value in y isin targeterrors[targetnumbers]
            if type(targeterror)==list:
                keeplist=targeterror+["IO"]
            else:
                keeplist=[targeterror,"IO"]
            
            data=data.loc[data["FEHLERART_BEZ"].isin(keeplist)]
            #targeterrors=targeterrors[:-1]
            #rename all y values that isin targeterrors[targetnumbers] in NIO
            if type(targeterror)==list:
                data["FEHLERART_BEZ"].replace(targeterrors[0],"NIO",inplace=True)
                keeplist=["NIO"]       
            else:
                keeplist=[targeterror]
            #balancing

            if balancing_trigger == True:
                balancing="balanced"
                unbalanced_data=data.copy()
                df_majority_IO = data[data["FEHLERART_BEZ"]=='IO']
                #df_majority_Blase = df_IO_Blase_poroes[df_IO_Blase_poroes['FEHLERKLASSE']=='Blase']
                df_minority_NIO = data.loc[data["FEHLERART_BEZ"].isin(keeplist)]
                # Downsample majority class
                df_majority_downsampled_IO = resample(df_majority_IO, 
                                                replace=False,    # sample without replacement
                                                n_samples=len(df_minority_NIO),     # to match minority class
                                                random_state=123) # reproducible results

                # Combine minority class with downsampled majority class
                y_all=data["FEHLERART_BEZ"]
                alldata=data.drop("FEHLERART_BEZ",axis=1)
                data = pd.concat([df_majority_downsampled_IO, df_minority_NIO])
                
                # Display new class counts
                print(data["FEHLERART_BEZ"].value_counts())
            else:
                balancing="unbalanced"
                y_all=data["FEHLERART_BEZ"]
                alldata=data.drop("FEHLERART_BEZ",axis=1)
            #X und Y wieder trennen
            y=data["FEHLERART_BEZ"].copy()
            data=data.drop("FEHLERART_BEZ",axis=1)

            for i in range(len(scalerlist)):
                scalername=scalernamelist[i]
                print("jetzt:",dataname,errorname,scalername)
                if NN_trigger == True:
                    y.replace(["0","1"],["IO",errorname],inplace=True)
                    y_all.replace(["0","1"],["IO",errorname],inplace=True)   
                    count_features=alldata.shape[1]
                    count_rows_all=alldata.shape[0]
                    count_rows_all_NIO=y_all[y_all == errorname].count()
                    count_rows_all_IO=y_all[y_all == "IO"].count()
                    count_rows_balanced=data.shape[0]
                    count_rows_balanced_NIO=y[y == errorname].count()
                    count_rows_balanced_IO=y[y == "IO"].count()                               
                    for l in range(len(NN_layers)):
                        for NN_activation_i,NN_solver_i,NN_alpha_i,NN_minibatch_size_i,NN_learning_rate_i,NN_learning_rate_init_i,NN_early_stopping_i,NN_tol_i,NN_n_iter_no_change_i,NN_validation_fraction_i,NN_max_fun_i,NN_max_iter_i,NN_beta_1_i,NN_beta_2_i,NN_epsilon_i,NN_momentum_i,NN_nesterovs_momentum_i,NN_power_t_i,NN_shuffle_i in product(NN_activation,NN_solver,NN_alpha,NN_minibatch_size,NN_learning_rate,NN_learning_rate_init,NN_early_stopping,NN_tol,NN_n_iter_no_change,NN_validation_fraction,NN_max_fun,NN_max_iter,NN_beta_1,NN_beta_2,NN_epsilon,NN_momentum,NN_nesterovs_momentum,NN_power_t,NN_shuffle):
                            best_trigger=False
                            count_IO_true_all_data_rand_list=[]
                            count_IO_false_all_data_rand_list=[]
                            count_NIO_false_all_data_rand_list=[]
                            count_NIO_true_all_data_rand_list=[]
                            count_IO_true_test_data_rand_list=[]
                            count_IO_false_test_data_rand_list=[]
                            count_NIO_false_test_data_rand_list=[]
                            count_NIO_true_test_data_rand_list=[]
                            count_NIO_test_rand_list=[]
                            count_IO_test_rand_list=[]
                            count_NIO_train_rand_list=[]
                            count_IO_train_rand_list=[]
                            balanced_acc_test_rand_list=[]
                            acc_test_rand_list=[]
                            balanced_acc_all_rand_list=[]
                            acc_all_rand_list=[]
                            precision_IO_all_rand_list=[]
                            recall_IO_all_rand_list=[]
                            fscore_IO_all_rand_list=[]
                            precision_NIO_all_rand_list=[]
                            recall_NIO_all_rand_list=[]
                            fscore_NIO_all_rand_list=[]
                            precision_IO_test_rand_list=[]
                            recall_IO_test_rand_list=[]
                            fscore_IO_test_rand_list=[]
                            precision_NIO_test_rand_list=[]
                            recall_NIO_test_rand_list=[]
                            fscore_NIO_test_rand_list=[]
                            modeltrainingloss_rand_list=[]
                            modelvalidationloss_rand_list=[]
                            losscurve_list=[]
                            losscurve_val_list=[]
                            if validation_trigger==True:
                                data_train, X_val_org, y_data_train, y_val = train_test_split(data, y, test_size =validation_fraction, random_state = randomseeds[0])
                                count_NIO_val=y_val[y_val == errorname].count()
                                count_IO_val=y_val[y_val == "IO"].count()     
                                balanced_acc_val_rand_list=[]
                                acc_val_rand_list=[]
                                count_IO_true_val_data_rand_list=[]
                                count_IO_false_val_data_rand_list=[]
                                count_NIO_false_val_data_rand_list=[]
                                count_NIO_true_val_data_rand_list=[]
                                count_NIO_val_rand_list=[]
                                count_IO_val_rand_list=[]    
                                precision_IO_val_rand_list=[]
                                recall_IO_val_rand_list=[]
                                fscore_IO_val_rand_list=[]
                                precision_NIO_val_rand_list=[]
                                recall_NIO_val_rand_list=[]
                                fscore_NIO_val_rand_list=[]                              
                            else:
                                data_train=data
                                y_data_train=y
    
                                    
                            for NN_random_state_i in randomseeds:
                                X_train, X_test, y_train, y_test = train_test_split(data_train, y_data_train, test_size =0.1, random_state = NN_random_state_i)

                                if scalernamelist[i]!="none":
                                    he_columnnames = list(X_train.filter(regex='_he', axis=1).columns)
                                    scaler = scalerlist[i]()
                                    scaler.fit(X_train)

                                    puffer=X_train[he_columnnames].copy().reset_index()
                                    X_train = pd.DataFrame(scaler.transform(X_train),columns=data.columns)
                                    X_train[he_columnnames]=puffer[he_columnnames]

                                    puffer=X_test[he_columnnames].copy().reset_index()
                                    X_test = pd.DataFrame(scaler.transform(X_test),columns=data.columns)
                                    X_test[he_columnnames]=puffer[he_columnnames]

                                    puffer=alldata[he_columnnames].copy().reset_index()
                                    X_all = pd.DataFrame(scaler.transform(alldata),columns=alldata.columns)
                                    X_all[he_columnnames]=puffer[he_columnnames]

                                    if validation_trigger==True:
                                        X_val=X_val_org.copy()
                                        puffer=X_val[he_columnnames].copy().reset_index()
                                        X_val = pd.DataFrame(scaler.transform(X_val),columns=data.columns)
                                        X_val[he_columnnames]=puffer[he_columnnames]                                
                                else:
                                    X_all=alldata
                                maxdivisor=1
                                if X_train.shape[0]>2*NN_minibatch_size_i:
                                    for m in primelist:                        
                                        if  X_train.shape[0]/m < NN_minibatch_size_i:
                                            break
                                        maxdivisor=m
                                        if ((X_train.shape[0]/m >= NN_minibatch_size_i) and (X_train.shape[0] % m)) == 0:
                                            maxdivisor=m
                                if X_train.shape[0] % maxdivisor == 0:
                                    NN_batch_size=int(X_train.shape[0]/maxdivisor)
                                else:
                                    NN_batch_size=int(math.ceil(X_train.shape[0]/maxdivisor))                                
                                print("Training of NN model:",runname,NN_layers_names[l],dataname,xgbfraction,balancing,errorname,scalername)
                            
                                NNmodel_rand=MLPClassifier(
                                    hidden_layer_sizes=NN_layers[l],
                                    activation=NN_activation_i,
                                    solver=NN_solver_i,
                                    alpha=NN_alpha_i,
                                    batch_size=NN_batch_size,
                                    learning_rate=NN_learning_rate_i,
                                    learning_rate_init=NN_learning_rate_init_i,
                                    power_t=NN_power_t_i,
                                    max_iter=NN_max_iter_i,
                                    shuffle=NN_shuffle_i,
                                    random_state=NN_random_state_i,
                                    tol=NN_tol_i,
                                    verbose=NN_verbose[0],
                                    early_stopping=NN_early_stopping_i,
                                    validation_fraction=NN_validation_fraction_i,
                                    beta_1=NN_beta_1_i,
                                    beta_2=NN_beta_2_i,
                                    epsilon=NN_epsilon_i,
                                    n_iter_no_change=NN_n_iter_no_change_i,
                                    max_fun=NN_max_fun_i,
                                    momentum=NN_momentum_i,
                                    nesterovs_momentum=NN_nesterovs_momentum_i

                                    )
                                NNmodel_rand.fit(X_train.values,y_train.values)
                                predicitions_alldata = NNmodel_rand.predict(X_all.values)
                                predicitions_testdata = NNmodel_rand.predict(X_test.values)
                                modeltrainingloss=NNmodel_rand.loss_curve_[-1]
                                modelvalidationloss=NNmodel_rand.validation_scores_[-1]    

                                losscurve_list=losscurve_list+[list(NNmodel_rand.loss_curve_)]
                                losscurve_val_list=losscurve_val_list+[list(NNmodel_rand.validation_scores_)]                 

                                classificationreport_alldata=classification_report(y_all,predicitions_alldata,zero_division=0)
                                precision_IO_all,recall_IO_all,fscore_IO_all = list(precision_recall_fscore_support(y_all,predicitions_alldata,pos_label="IO",average="binary",zero_division=0))[:3]
                                precision_NIO_all,recall_NIO_all,fscore_NIO_all = list(precision_recall_fscore_support(y_all,predicitions_alldata,pos_label=errorname,average="binary",zero_division=0))[:3]
                                precision_IO_test,recall_IO_test,fscore_IO_test = list(precision_recall_fscore_support(y_test,predicitions_testdata,pos_label="IO",average="binary",zero_division=0))[:3]
                                precision_NIO_test,recall_NIO_test,fscore_NIO_test = list(precision_recall_fscore_support(y_test,predicitions_testdata,pos_label=errorname,average="binary",zero_division=0))[:3]
                                
                                balanced_acc_test=balanced_accuracy_score(y_test,predicitions_testdata)
                                acc_test=accuracy_score(y_test,predicitions_testdata)
                                balanced_acc_all=balanced_accuracy_score(y_all,predicitions_alldata)
                                acc_all=accuracy_score(y_all,predicitions_alldata)

                                count_NIO_test=y_test[y_test == errorname].count()
                                count_IO_test=y_test[y_test == "IO"].count()
                                count_NIO_train=y_train[y_train == errorname].count()
                                count_IO_train=y_train[y_train == "IO"].count()
                                

                                confusionmatrix_alldata=confusion_matrix(y_all,predicitions_alldata,labels=["IO",errorname])
                                count_IO_true_all_data,count_NIO_false_all_data,count_IO_false_all_data,count_NIO_true_all_data=[confusionmatrix_alldata[0][0],confusionmatrix_alldata[0][1],confusionmatrix_alldata[1][0],confusionmatrix_alldata[1][1]]
                                confusionmatrix_testdata=confusion_matrix(y_test,predicitions_testdata,labels=["IO",errorname])
                                count_IO_true_test_data,count_NIO_false_test_data,count_IO_false_test_data,count_NIO_true_test_data=[confusionmatrix_testdata[0][0],confusionmatrix_testdata[0][1],confusionmatrix_testdata[1][0],confusionmatrix_testdata[1][1]]

                                count_IO_true_all_data_rand_list=count_IO_true_all_data_rand_list+[count_IO_true_all_data]
                                count_IO_false_all_data_rand_list=count_IO_false_all_data_rand_list+[count_IO_false_all_data]
                                count_NIO_false_all_data_rand_list=count_NIO_false_all_data_rand_list+[count_NIO_false_all_data]
                                count_NIO_true_all_data_rand_list=count_NIO_true_all_data_rand_list+[count_NIO_true_all_data]
                                count_IO_true_test_data_rand_list=count_IO_true_test_data_rand_list+[count_IO_true_test_data]
                                count_IO_false_test_data_rand_list=count_IO_false_test_data_rand_list+[count_IO_false_test_data]
                                count_NIO_false_test_data_rand_list=count_NIO_false_test_data_rand_list+[count_NIO_false_test_data]
                                count_NIO_true_test_data_rand_list=count_NIO_true_test_data_rand_list+[count_NIO_true_test_data]
                                count_NIO_test_rand_list=count_NIO_test_rand_list+[count_NIO_test]
                                count_IO_test_rand_list=count_IO_test_rand_list+[count_IO_test]
                                count_NIO_train_rand_list=count_NIO_train_rand_list+[count_NIO_train]
                                count_IO_train_rand_list=count_IO_train_rand_list+[count_IO_train]
                                balanced_acc_test_rand_list=balanced_acc_test_rand_list+[balanced_acc_test]
                                acc_test_rand_list=acc_test_rand_list+[acc_test]
                                balanced_acc_all_rand_list=balanced_acc_all_rand_list+[balanced_acc_all]
                                acc_all_rand_list=acc_all_rand_list+[acc_all]
                                precision_IO_all_rand_list=precision_IO_all_rand_list+[precision_IO_all]
                                recall_IO_all_rand_list=recall_IO_all_rand_list+[recall_IO_all]
                                fscore_IO_all_rand_list=fscore_IO_all_rand_list+[fscore_IO_all]
                                precision_NIO_all_rand_list=precision_NIO_all_rand_list+[precision_NIO_all]
                                recall_NIO_all_rand_list=recall_NIO_all_rand_list+[recall_NIO_all]
                                fscore_NIO_all_rand_list=fscore_NIO_all_rand_list+[fscore_NIO_all]
                                precision_IO_test_rand_list=precision_IO_test_rand_list+[precision_IO_test]
                                recall_IO_test_rand_list=recall_IO_test_rand_list+[recall_IO_test]
                                fscore_IO_test_rand_list=fscore_IO_test_rand_list+[fscore_IO_test]
                                precision_NIO_test_rand_list=precision_NIO_test_rand_list+[precision_NIO_test]
                                recall_NIO_test_rand_list=recall_NIO_test_rand_list+[recall_NIO_test]
                                fscore_NIO_test_rand_list=fscore_NIO_test_rand_list+[fscore_NIO_test]
                                modeltrainingloss_rand_list=modeltrainingloss_rand_list+[modeltrainingloss]
                                modelvalidationloss_rand_list=modelvalidationloss_rand_list+[modelvalidationloss]

                                if validation_trigger==True:
                                    predicitions_valdata = NNmodel_rand.predict(X_val.values)
                                    precision_NIO_val,recall_NIO_val,fscore_NIO_val = list(precision_recall_fscore_support(y_val,predicitions_valdata,pos_label=errorname,average="binary",zero_division=0))[:3]
                                    precision_IO_val,recall_IO_val,fscore_IO_val = list(precision_recall_fscore_support(y_val,predicitions_valdata,pos_label="IO",average="binary",zero_division=0))[:3]
                                    balanced_acc_val=balanced_accuracy_score(y_val,predicitions_valdata)
                                    acc_val=accuracy_score(y_val,predicitions_valdata)
                                    confusionmatrix_valdata=confusion_matrix(y_val,predicitions_valdata,labels=["IO",errorname])
                                    count_IO_true_val_data,count_NIO_false_val_data,count_IO_false_val_data,count_NIO_true_val_data=[confusionmatrix_valdata[0][0],confusionmatrix_valdata[0][1],confusionmatrix_valdata[1][0],confusionmatrix_valdata[1][1]]
                                    balanced_acc_val_rand_list=balanced_acc_val_rand_list+[balanced_acc_val]
                                    acc_val_rand_list=acc_val_rand_list+[acc_val]
                                    count_IO_true_val_data_rand_list=count_IO_true_val_data_rand_list+[count_IO_true_val_data]
                                    count_IO_false_val_data_rand_list=count_IO_false_val_data_rand_list+[count_IO_false_val_data]
                                    count_NIO_false_val_data_rand_list=count_NIO_false_val_data_rand_list+[count_NIO_false_val_data]
                                    count_NIO_true_val_data_rand_list=count_NIO_true_val_data_rand_list+[count_NIO_true_val_data]
                                    count_NIO_val_rand_list=count_NIO_val_rand_list+[count_NIO_val]
                                    count_IO_val_rand_list=count_IO_val_rand_list+[count_IO_val]
                                    precision_IO_val_rand_list=precision_IO_val_rand_list+[precision_IO_val]
                                    recall_IO_val_rand_list=recall_IO_val_rand_list+[recall_IO_val]
                                    fscore_IO_val_rand_list=fscore_IO_val_rand_list+[fscore_IO_val]
                                    precision_NIO_val_rand_list=precision_NIO_val_rand_list+[precision_NIO_val]
                                    recall_NIO_val_rand_list=recall_NIO_val_rand_list+[recall_NIO_val]
                                    fscore_NIO_val_rand_list=fscore_NIO_val_rand_list+[fscore_NIO_val]


                                if balanced_acc_test >= max(balanced_acc_test_rand_list):
                                    NNmodel=NNmodel_rand
                                    count_IO_true_all_data_best=count_IO_true_all_data
                                    count_IO_false_all_data_best=count_IO_false_all_data
                                    count_NIO_false_all_data_best=count_NIO_false_all_data
                                    count_NIO_true_all_data_best=count_NIO_true_all_data
                                    count_IO_true_test_data_best=count_IO_true_test_data
                                    count_IO_false_test_data_best=count_IO_false_test_data
                                    count_NIO_false_test_data_best=count_NIO_false_test_data
                                    count_NIO_true_test_data_best=count_NIO_true_test_data
                                    count_NIO_test_best=count_NIO_test
                                    count_IO_test_best=count_IO_test
                                    count_NIO_train_best=count_NIO_train
                                    count_IO_train_best=count_IO_train
                                    balanced_acc_test_best=balanced_acc_test
                                    acc_test_best=acc_test
                                    balanced_acc_all_best=balanced_acc_all
                                    acc_all_best=acc_all
                                    precision_IO_all_best=precision_IO_all
                                    recall_IO_all_best=recall_IO_all
                                    fscore_IO_all_best=fscore_IO_all
                                    precision_NIO_all_best=precision_NIO_all
                                    recall_NIO_all_best=recall_NIO_all
                                    fscore_NIO_all_best=fscore_NIO_all
                                    precision_IO_test_best=precision_IO_test
                                    recall_IO_test_best=recall_IO_test
                                    fscore_IO_test_best=fscore_IO_test
                                    precision_NIO_test_best=precision_NIO_test
                                    recall_NIO_test_best=recall_NIO_test
                                    fscore_NIO_test_best=fscore_NIO_test
                                    modeltrainingloss_best=modeltrainingloss
                                    modelvalidationloss_best=modelvalidationloss
                                    if validation_trigger==True:
                                        balanced_acc_val_best=balanced_acc_val
                                        acc_val_best=acc_val
                                        count_IO_true_val_data_best=count_IO_true_val_data
                                        count_IO_false_val_data_best=count_IO_false_val_data
                                        count_NIO_false_val_data_best=count_NIO_false_val_data
                                        count_NIO_true_val_data_best=count_NIO_true_val_data
                                        count_NIO_val_best=count_NIO_val
                                        count_IO_val_best=count_IO_val
                                        precision_IO_val_best=precision_IO_val
                                        recall_IO_val_best=recall_IO_val
                                        fscore_IO_val_best=fscore_IO_val
                                        precision_NIO_val_best=precision_NIO_val
                                        recall_NIO_val_best=recall_NIO_val
                                        fscore_NIO_val_best=fscore_NIO_val


                            count_IO_true_all_data=np.mean(count_IO_true_all_data_rand_list)
                            count_IO_false_all_data=np.mean(count_IO_false_all_data_rand_list)
                            count_NIO_false_all_data=np.mean(count_NIO_false_all_data_rand_list)
                            count_NIO_true_all_data=np.mean(count_NIO_true_all_data_rand_list)
                            count_IO_true_test_data=np.mean(count_IO_true_test_data_rand_list)
                            count_IO_false_test_data=np.mean(count_IO_false_test_data_rand_list)
                            count_NIO_false_test_data=np.mean(count_NIO_false_test_data_rand_list)
                            count_NIO_true_test_data=np.mean(count_NIO_true_test_data_rand_list)
                            count_NIO_test=np.mean(count_NIO_test_rand_list)
                            count_IO_test=np.mean(count_IO_test_rand_list)
                            count_NIO_train=np.mean(count_NIO_train_rand_list)
                            count_IO_train=np.mean(count_IO_train_rand_list)
                            balanced_acc_test=np.mean(balanced_acc_test_rand_list)
                            acc_test=np.mean(acc_test_rand_list)
                            balanced_acc_all=np.mean(balanced_acc_all_rand_list)
                            acc_all=np.mean(acc_all_rand_list)
                            precision_IO_all=np.mean(precision_IO_all_rand_list)
                            recall_IO_all=np.mean(recall_IO_all_rand_list)
                            fscore_IO_all=np.mean(fscore_IO_all_rand_list)
                            precision_NIO_all=np.mean(precision_NIO_all_rand_list)
                            recall_NIO_all=np.mean(recall_NIO_all_rand_list)
                            fscore_NIO_all=np.mean(fscore_NIO_all_rand_list)
                            precision_IO_test=np.mean(precision_IO_test_rand_list)
                            recall_IO_test=np.mean(recall_IO_test_rand_list)
                            fscore_IO_test=np.mean(fscore_IO_test_rand_list)
                            precision_NIO_test=np.mean(precision_NIO_test_rand_list)
                            recall_NIO_test=np.mean(recall_NIO_test_rand_list)
                            fscore_NIO_test=np.mean(fscore_NIO_test_rand_list)
                            modeltrainingloss=np.mean(modeltrainingloss_rand_list)
                            modelvalidationloss=np.mean(modelvalidationloss_rand_list)
                            balanced_acc_val=np.mean(balanced_acc_val_rand_list)
                            acc_val=np.mean(acc_val_rand_list)
                            count_IO_true_val_data=np.mean(count_IO_true_val_data_rand_list)
                            count_IO_false_val_data=np.mean(count_IO_false_val_data_rand_list)
                            count_NIO_false_val_data=np.mean(count_NIO_false_val_data_rand_list)
                            count_NIO_true_val_data=np.mean(count_NIO_true_val_data_rand_list)
                            count_NIO_val=np.mean(count_NIO_val_rand_list)
                            count_IO_val=np.mean(count_IO_val_rand_list)
                            precision_IO_val=np.mean(precision_IO_val_rand_list)
                            recall_IO_val=np.mean(recall_IO_val_rand_list)
                            fscore_IO_val=np.mean(fscore_IO_val_rand_list)
                            precision_NIO_val=np.mean(precision_NIO_val_rand_list)
                            recall_NIO_val=np.mean(recall_NIO_val_rand_list)
                            fscore_NIO_val=np.mean(fscore_NIO_val_rand_list)


                            scores=[
                                ["NN"]+[runname]+[dataname]+[xgbfraction]+[errorname]+[balancing]+[scalername]+[randomseeds_str]+
                                
                                [count_features]+[count_rows_all]+
                                [count_rows_balanced]+[count_rows_balanced_IO]+[count_rows_balanced_NIO]+
                                
                                [count_IO_test]+[count_NIO_test]+
                                [balanced_acc_test]+[precision_NIO_test]+[recall_NIO_test]+[fscore_NIO_test]+[precision_IO_test]+[recall_IO_test]+[fscore_IO_test]+[acc_test]+
                                [count_IO_test_best]+[count_NIO_test_best]+
                                [balanced_acc_test_best]+[precision_NIO_test_best]+[recall_NIO_test_best]+[fscore_NIO_test_best]+[precision_IO_test_best]+[recall_IO_test_best]+[fscore_IO_test_best]+[acc_test_best]+                           

                                [count_rows_all_IO]+[count_rows_all_NIO]+
                                [balanced_acc_all]+[precision_NIO_all]+[recall_NIO_all]+[fscore_NIO_all]+[precision_IO_all]+[recall_IO_all]+[fscore_IO_all]+[acc_all]+
                                [balanced_acc_all_best]+[precision_NIO_all_best]+[recall_NIO_all_best]+[fscore_NIO_all_best]+[precision_IO_all_best]+[recall_IO_all_best]+[fscore_IO_all_best]+[acc_all_best]+    

                                [count_IO_val]+[count_NIO_val]+
                                [balanced_acc_val]+[precision_NIO_val]+[recall_NIO_val]+[fscore_NIO_val]+[precision_IO_val]+[recall_IO_val]+[fscore_IO_val]+[acc_val]+
                                [balanced_acc_val_best]+[precision_NIO_val_best]+[recall_NIO_val_best]+[fscore_NIO_val_best]+[precision_IO_val_best]+[recall_IO_val_best]+[fscore_IO_val_best]+[acc_val_best]+
                            
                                [count_IO_true_test_data]+[count_IO_false_test_data]+[count_NIO_false_test_data]+[count_NIO_true_test_data]+
                                [count_IO_true_test_data_best]+[count_IO_false_test_data_best]+[count_NIO_false_test_data_best]+[count_NIO_true_test_data_best]+ 
                                [count_IO_true_val_data]+[count_IO_false_val_data]+[count_NIO_false_val_data]+[count_NIO_true_val_data]+
                                [count_IO_true_val_data_best]+[count_IO_false_val_data_best]+[count_NIO_false_val_data_best]+[count_NIO_true_val_data_best]+
                                [count_IO_true_all_data]+[count_IO_false_all_data]+[count_NIO_false_all_data]+[count_NIO_true_all_data]+
                                [count_IO_true_all_data_best]+[count_IO_false_all_data_best]+[count_NIO_false_all_data_best]+[count_NIO_true_all_data_best]+
                            
                                [count_IO_train]+[count_NIO_train]+[count_IO_train_best]+[count_NIO_train_best]+

                                [modeltrainingloss]+[modelvalidationloss]+
                                [modeltrainingloss_best]+[modelvalidationloss_best]+          

                                [NN_solver_i]+[NN_activation_i]+[NN_layers_names[l]]+[NN_learning_rate_init_i]+[NN_alpha_i]+[NN_minibatch_size_i]+[NN_max_iter_i]+[NN_n_iter_no_change_i]+[NN_tol_i]+
                                [NN_validation_fraction_i]+[NN_beta_1_i]+[NN_beta_2_i]+[NN_epsilon_i]+[NN_max_fun_i]+[NN_learning_rate_i]+[NN_shuffle_i]+[NN_early_stopping_i]+[NN_power_t_i]+[NN_momentum_i]+[NN_nesterovs_momentum_i]+
                                ["none"]+["none"]+["none"]+["none"]+["none"]+["none"]+["none"]+["none"]+["none"]+ ["none"]+["none"]+["none"]+["none"]+["none"]+["none"]
                            ]
                            if best_export_trigger == True:
                                entryname=str(dataname)+"_"+str(xgbfraction).zfill(3)+"_"+errorname+"_NN"
                                if balanced_acc_test > best_dict[entryname]:
                                    best_dict[entryname]=balanced_acc_test.copy()
                                    best_score_dict[entryname+"_scorelist"]=scores[0].copy()
                                    exportdata=pd.DataFrame.from_dict(best_score_dict,orient='index',columns=columnnames)
                                    with open(savepath+r"\\best_scores.csv", mode='w', newline='\n') as f:
                                        exportdata.to_csv(f, float_format='%.6f',index=False) 
                                    best_trigger=True     

                            globalscores=globalscores+scores
                            
                            losscurve_length=len(losscurve_list[0])
                            for losscurve_i in losscurve_list:
                                losscurve_length=min(losscurve_length,len(losscurve_i))
                            losscurve_list_new=[]

                            
                            for losscurve_i in losscurve_list:
                                losscurve_list_new=losscurve_list_new+[losscurve_i[:losscurve_length]]

                            losscurve_array=np.stack(losscurve_list_new,axis=0)
                            losscurve=np.mean(losscurve_array,axis=0)

                            losscurve_val_length=len(losscurve_val_list[0])
                            for losscurve_i in losscurve_val_list:

                                losscurve_val_length=min(losscurve_val_length,len(losscurve_i))

                            losscurve_val_list_new=[]
                            
                            for losscurve_i in losscurve_val_list:
                                losscurve_val_list_new=losscurve_val_list_new+[losscurve_i[:losscurve_val_length]]
                                
                            losscurve_val_array=np.stack(losscurve_val_list_new,axis=0)
                            losscurve_val=np.mean(losscurve_val_array,axis=0)

                            pl.figure(figsize=(1, 1), dpi=80)
                            pl.plot(losscurve,label='training')
                            pl.plot(losscurve_val,label='validation')
                            pl.xlabel('iteration')
                            pl.ylabel('loss')
                            pl.title("losscurves_avg_"+NN_layers_names[l]+"_"+entryname[:-4])
                            pl.legend()
                            fig = pl.gcf()
                            fig.set_size_inches(18.5, 10.5)
                            fig.tight_layout()
                            str_time=datetime.now().strftime('%Y%m%d-%H%M%S') 
                            if best_trigger == True:
                                with open(savepath+r"\\"+entryname+"_avg_losscurve.eps", mode='w') as f:
                                    pl.savefig(f,bbox_inches='tight', format='eps')
                                with open(savepath+r"\\"+entryname+"_avg_losscurve.png", mode='wb') as f:
                                    pl.savefig(f,bbox_inches='tight', format='png')                      
                            pl.close(True)
                            
                            pl.figure(figsize=(1, 1), dpi=80)
                            pl.plot(list(NNmodel.loss_curve_),label='training')
                            pl.plot(list(NNmodel.validation_scores_),label='validation')
                            pl.xlabel('iteration')
                            pl.ylabel('loss')
                            pl.title("losscurves_best_"+NN_layers_names[l]+"_"+entryname[:-4])
                            pl.legend()
                            fig = pl.gcf()
                            fig.set_size_inches(18.5, 10.5)
                            fig.tight_layout()
                            str_time=datetime.now().strftime('%Y%m%d-%H%M%S') 
                            if best_trigger == True:
                                with open(savepath+r"\\"+entryname+"_best_losscurve.eps", mode='w') as f:
                                    pl.savefig(f,bbox_inches='tight', format='eps')
                                with open(savepath+r"\\"+entryname+"_best_losscurve.png", mode='wb') as f:
                                    pl.savefig(f,bbox_inches='tight', format='png')   
                                with open(savepath+r"\\"+entryname+"_bestmodel", mode='wb') as f:
                                    joblib.dump(NNmodel, f)                           
                            pl.close(True)                            
                            print("Ende NN")
                            pd_dataset=pd.DataFrame(columns=columnnames,data=globalscores)
                            str_time=datetime.now().strftime('%Y%m%d-%H%M%S')
                            with open(savepath+r"\\scores_saveslot_"+str(saveslotcounter)+".csv", mode='w', newline='\n') as f:
                                pd_dataset.to_csv(f, float_format='%.6f',index=True) 
                            saveslotcounter=saveslotcounter+1
                            if saveslotcounter>10:
                                saveslotcounter=1
                
                if XGB_trigger == True:
                    y.replace(["IO",errorname],[0,1],inplace=True)
                    y_all.replace(["IO",errorname],[0,1],inplace=True)  
                    count_features=alldata.shape[1]
                    count_rows_all=alldata.shape[0]
                    count_rows_all_NIO=y_all[y_all == 1].count()
                    count_rows_all_IO=y_all[y_all == 0].count()
                    count_rows_balanced=data.shape[0]
                    count_rows_balanced_NIO=y[y == 1].count()
                    count_rows_balanced_IO=y[y == 0].count()                  
                    for XGB_objective_i,XGB_seed_i,XGB_eta_i,XGB_alpha_i,XGB_gamma_i,XGB_lambda_i,XGB_minchild_weight_i,XGB_max_delta_step_i,XGB_n_estimators_i,XGB_earlystoppingrounds_i,XGB_parallel_tree_i,XGB_maxdepth_i,XGB_subsample_i,XGB_colsample_bytree_i,XGB_colsample_bylevel_i,XGB_colsample_bynode_i in product(XGB_objective,XGB_seed,XGB_eta,XGB_alpha,XGB_gamma,XGB_lambda,XGB_minchild_weight,XGB_max_delta_step,XGB_n_estimators,XGB_earlystoppingrounds,XGB_parallel_tree,XGB_maxdepth,XGB_subsample,XGB_colsample_bytree,XGB_colsample_bylevel,XGB_colsample_bynode):
                        best_trigger=False
                        count_IO_true_all_data_rand_list=[]
                        count_IO_false_all_data_rand_list=[]
                        count_NIO_false_all_data_rand_list=[]
                        count_NIO_true_all_data_rand_list=[]
                        count_IO_true_test_data_rand_list=[]
                        count_IO_false_test_data_rand_list=[]
                        count_NIO_false_test_data_rand_list=[]
                        count_NIO_true_test_data_rand_list=[]
                        count_NIO_test_rand_list=[]
                        count_IO_test_rand_list=[]
                        count_NIO_train_rand_list=[]
                        count_IO_train_rand_list=[]
                        balanced_acc_test_rand_list=[]
                        acc_test_rand_list=[]
                        balanced_acc_all_rand_list=[]
                        acc_all_rand_list=[]
                        precision_IO_all_rand_list=[]
                        recall_IO_all_rand_list=[]
                        fscore_IO_all_rand_list=[]
                        precision_NIO_all_rand_list=[]
                        recall_NIO_all_rand_list=[]
                        fscore_NIO_all_rand_list=[]
                        precision_IO_test_rand_list=[]
                        recall_IO_test_rand_list=[]
                        fscore_IO_test_rand_list=[]
                        precision_NIO_test_rand_list=[]
                        recall_NIO_test_rand_list=[]
                        fscore_NIO_test_rand_list=[]
                        modeltrainingloss_rand_list=[]
                        modelvalidationloss_rand_list=[]
                        shap_values_list=[]
                        expected_value_list=[]
                        losscurve_list=[]
                        losscurve_val_list=[]

                        if validation_trigger==True:
                            data_train, X_val_org, y_data_train, y_val = train_test_split(data, y, test_size =validation_fraction, random_state = randomseeds[0])
                            count_NIO_val=y_val[y_val == 1].count()
                            count_IO_val=y_val[y_val == 0].count()     
                            balanced_acc_val_rand_list=[]
                            acc_val_rand_list=[]
                            count_IO_true_val_data_rand_list=[]
                            count_IO_false_val_data_rand_list=[]
                            count_NIO_false_val_data_rand_list=[]
                            count_NIO_true_val_data_rand_list=[]
                            count_NIO_val_rand_list=[]
                            count_IO_val_rand_list=[]    
                            precision_IO_val_rand_list=[]
                            recall_IO_val_rand_list=[]
                            fscore_IO_val_rand_list=[]
                            precision_NIO_val_rand_list=[]
                            recall_NIO_val_rand_list=[]
                            fscore_NIO_val_rand_list=[]                              
                        else:
                            data_train=data
                            y_data_train=y
                        for XGB_seed_i in randomseeds:
                            X_train, X_test, y_train, y_test = train_test_split(data, y, test_size =0.1, random_state = XGB_seed_i)
                            if scalernamelist[i]!="none":
                                he_columnnames = list(X_train.filter(regex='_he', axis=1).columns)
                                scaler = scalerlist[i]()
                                scaler.fit(X_train)

                                puffer=X_train[he_columnnames].copy().reset_index()
                                X_train = pd.DataFrame(scaler.transform(X_train),columns=data.columns)
                                X_train[he_columnnames]=puffer[he_columnnames]

                                puffer=X_test[he_columnnames].copy().reset_index()
                                X_test = pd.DataFrame(scaler.transform(X_test),columns=data.columns)
                                X_test[he_columnnames]=puffer[he_columnnames]

                                puffer=alldata[he_columnnames].copy().reset_index()
                                X_all = pd.DataFrame(scaler.transform(alldata),columns=alldata.columns)
                                X_all[he_columnnames]=puffer[he_columnnames]

                                if validation_trigger==True:
                                    X_val=X_val_org.copy()
                                    puffer=X_val[he_columnnames].copy().reset_index()
                                    X_val = pd.DataFrame(scaler.transform(X_val),columns=data.columns)
                                    X_val[he_columnnames]=puffer[he_columnnames]   
                            else:
                                X_all=alldata
                            print("Training of XGB model:",runname,dataname,balancing,errorname,scalername)
                            XGBmodel_rand = xgb.XGBClassifier(max_depth=XGB_maxdepth_i,
                                                learning_rate=XGB_eta_i, 
                                                n_estimators=XGB_n_estimators_i,
                                                verbosity=XGB_verbosity[0],
                                                objective=XGB_objective_i,
                                                gamma=XGB_gamma_i,
                                                min_child_weight=XGB_minchild_weight_i,
                                                max_delta_step=XGB_max_delta_step_i, 
                                                subsample=XGB_subsample_i,
                                                colsample_bytree=XGB_colsample_bytree_i, 
                                                colsample_bylevel=XGB_colsample_bylevel_i,
                                                colsample_bynode=XGB_colsample_bynode_i, 
                                                reg_alpha=XGB_alpha_i, 
                                                reg_lambda=XGB_lambda_i,
                                                random_state=XGB_seed_i,
                                                num_parallel_tree=XGB_parallel_tree_i,
                                                importance_type="gain",
                                                early_stopping_round=XGB_earlystoppingrounds_i
                                                )
                            XGBmodel_rand.fit(X_train,y_train,eval_set = [(X_train, y_train), (X_test,y_test)],early_stopping_rounds=XGB_earlystoppingrounds_i)
                            XGBlossplotdata = XGBmodel_rand.evals_result()
                            predicitions_alldata = XGBmodel_rand.predict(X_all.values)
                            predicitions_testdata = XGBmodel_rand.predict(X_test.values)
                            explainer = shap.TreeExplainer(XGBmodel_rand)
                            shap_values = explainer.shap_values(X_all.values)
                            expected_value = explainer.expected_value
                            shap_values_list=shap_values_list+[shap_values]
                            expected_value_list=expected_value_list+[expected_value]
                            
                            # plot learning curves

                            modeltrainingloss=XGBlossplotdata['validation_0']['logloss'][-1]
                            modelvalidationloss=XGBlossplotdata['validation_1']['logloss'][-1]

                            losscurve_list=losscurve_list+[list(XGBlossplotdata['validation_0']['logloss'])]
                            losscurve_val_list=losscurve_val_list+[list(XGBlossplotdata['validation_1']['logloss'])]

                            classificationreport_alldata=classification_report(y_all,predicitions_alldata,zero_division=0)
                            precision_IO_all,recall_IO_all,fscore_IO_all = list(precision_recall_fscore_support(y_all,predicitions_alldata,pos_label=0,average="binary",zero_division=0))[:3]
                            precision_NIO_all,recall_NIO_all,fscore_NIO_all = list(precision_recall_fscore_support(y_all,predicitions_alldata,pos_label=1,average="binary",zero_division=0))[:3]
                            precision_IO_test,recall_IO_test,fscore_IO_test = list(precision_recall_fscore_support(y_test,predicitions_testdata,pos_label=0,average="binary",zero_division=0))[:3]
                            precision_NIO_test,recall_NIO_test,fscore_NIO_test = list(precision_recall_fscore_support(y_test,predicitions_testdata,pos_label=1,average="binary",zero_division=0))[:3]
                                    
                            balanced_acc_test=balanced_accuracy_score(y_test,predicitions_testdata)
                            acc_test=accuracy_score(y_test,predicitions_testdata)
                            balanced_acc_all=balanced_accuracy_score(y_all,predicitions_alldata)
                            acc_all=accuracy_score(y_all,predicitions_alldata)

                            count_NIO_test=y_test[y_test == 1].count()
                            count_IO_test=y_test[y_test == 0].count()
                            count_NIO_train=y_train[y_train == 1].count()
                            count_IO_train=y_train[y_train == 0].count()
                            
                            confusionmatrix_alldata=confusion_matrix(y_all,predicitions_alldata,labels=[0,1])
                            count_IO_true_all_data,count_NIO_false_all_data,count_IO_false_all_data,count_NIO_true_all_data=[confusionmatrix_alldata[0][0],confusionmatrix_alldata[0][1],confusionmatrix_alldata[1][0],confusionmatrix_alldata[1][1]]
                            confusionmatrix_testdata=confusion_matrix(y_test,predicitions_testdata,labels=[0,1])
                            count_IO_true_test_data,count_NIO_false_test_data,count_IO_false_test_data,count_NIO_true_test_data=[confusionmatrix_testdata[0][0],confusionmatrix_testdata[0][1],confusionmatrix_testdata[1][0],confusionmatrix_testdata[1][1]]

                            count_IO_true_all_data_rand_list=count_IO_true_all_data_rand_list+[count_IO_true_all_data]
                            count_IO_false_all_data_rand_list=count_IO_false_all_data_rand_list+[count_IO_false_all_data]
                            count_NIO_false_all_data_rand_list=count_NIO_false_all_data_rand_list+[count_NIO_false_all_data]
                            count_NIO_true_all_data_rand_list=count_NIO_true_all_data_rand_list+[count_NIO_true_all_data]
                            count_IO_true_test_data_rand_list=count_IO_true_test_data_rand_list+[count_IO_true_test_data]
                            count_IO_false_test_data_rand_list=count_IO_false_test_data_rand_list+[count_IO_false_test_data]
                            count_NIO_false_test_data_rand_list=count_NIO_false_test_data_rand_list+[count_NIO_false_test_data]
                            count_NIO_true_test_data_rand_list=count_NIO_true_test_data_rand_list+[count_NIO_true_test_data]
                            count_NIO_test_rand_list=count_NIO_test_rand_list+[count_NIO_test]
                            count_IO_test_rand_list=count_IO_test_rand_list+[count_IO_test]
                            count_NIO_train_rand_list=count_NIO_train_rand_list+[count_NIO_train]
                            count_IO_train_rand_list=count_IO_train_rand_list+[count_IO_train]
                            balanced_acc_test_rand_list=balanced_acc_test_rand_list+[balanced_acc_test]
                            acc_test_rand_list=acc_test_rand_list+[acc_test]
                            balanced_acc_all_rand_list=balanced_acc_all_rand_list+[balanced_acc_all]
                            acc_all_rand_list=acc_all_rand_list+[acc_all]
                            precision_IO_all_rand_list=precision_IO_all_rand_list+[precision_IO_all]
                            recall_IO_all_rand_list=recall_IO_all_rand_list+[recall_IO_all]
                            fscore_IO_all_rand_list=fscore_IO_all_rand_list+[fscore_IO_all]
                            precision_NIO_all_rand_list=precision_NIO_all_rand_list+[precision_NIO_all]
                            recall_NIO_all_rand_list=recall_NIO_all_rand_list+[recall_NIO_all]
                            fscore_NIO_all_rand_list=fscore_NIO_all_rand_list+[fscore_NIO_all]
                            precision_IO_test_rand_list=precision_IO_test_rand_list+[precision_IO_test]
                            recall_IO_test_rand_list=recall_IO_test_rand_list+[recall_IO_test]
                            fscore_IO_test_rand_list=fscore_IO_test_rand_list+[fscore_IO_test]
                            precision_NIO_test_rand_list=precision_NIO_test_rand_list+[precision_NIO_test]
                            recall_NIO_test_rand_list=recall_NIO_test_rand_list+[recall_NIO_test]
                            fscore_NIO_test_rand_list=fscore_NIO_test_rand_list+[fscore_NIO_test]
                            modeltrainingloss_rand_list=modeltrainingloss_rand_list+[modeltrainingloss]
                            modelvalidationloss_rand_list=modelvalidationloss_rand_list+[modelvalidationloss]

                            if validation_trigger==True:
                                predicitions_valdata = XGBmodel_rand.predict(X_val.values)
                                precision_NIO_val,recall_NIO_val,fscore_NIO_val = list(precision_recall_fscore_support(y_val,predicitions_valdata,pos_label=1,average="binary",zero_division=0))[:3]
                                precision_IO_val,recall_IO_val,fscore_IO_val = list(precision_recall_fscore_support(y_val,predicitions_valdata,pos_label=0,average="binary",zero_division=0))[:3]
                                balanced_acc_val=balanced_accuracy_score(y_val,predicitions_valdata)
                                acc_val=accuracy_score(y_val,predicitions_valdata)
                                confusionmatrix_valdata=confusion_matrix(y_val,predicitions_valdata,labels=[0,1])
                                count_IO_true_val_data,count_NIO_false_val_data,count_IO_false_val_data,count_NIO_true_val_data=[confusionmatrix_valdata[0][0],confusionmatrix_valdata[0][1],confusionmatrix_valdata[1][0],confusionmatrix_valdata[1][1]]
                                balanced_acc_val_rand_list=balanced_acc_val_rand_list+[balanced_acc_val]
                                acc_val_rand_list=acc_val_rand_list+[acc_val]
                                count_IO_true_val_data_rand_list=count_IO_true_val_data_rand_list+[count_IO_true_val_data]
                                count_IO_false_val_data_rand_list=count_IO_false_val_data_rand_list+[count_IO_false_val_data]
                                count_NIO_false_val_data_rand_list=count_NIO_false_val_data_rand_list+[count_NIO_false_val_data]
                                count_NIO_true_val_data_rand_list=count_NIO_true_val_data_rand_list+[count_NIO_true_val_data]
                                count_NIO_val_rand_list=count_NIO_val_rand_list+[count_NIO_val]
                                count_IO_val_rand_list=count_IO_val_rand_list+[count_IO_val]
                                precision_IO_val_rand_list=precision_IO_val_rand_list+[precision_IO_val]
                                recall_IO_val_rand_list=recall_IO_val_rand_list+[recall_IO_val]
                                fscore_IO_val_rand_list=fscore_IO_val_rand_list+[fscore_IO_val]
                                precision_NIO_val_rand_list=precision_NIO_val_rand_list+[precision_NIO_val]
                                recall_NIO_val_rand_list=recall_NIO_val_rand_list+[recall_NIO_val]
                                fscore_NIO_val_rand_list=fscore_NIO_val_rand_list+[fscore_NIO_val]

                            if balanced_acc_test >= max(balanced_acc_test_rand_list):
                                XGBmodel=XGBmodel_rand
                                count_IO_true_all_data_best=count_IO_true_all_data
                                count_IO_false_all_data_best=count_IO_false_all_data
                                count_NIO_false_all_data_best=count_NIO_false_all_data
                                count_NIO_true_all_data_best=count_NIO_true_all_data
                                count_IO_true_test_data_best=count_IO_true_test_data
                                count_IO_false_test_data_best=count_IO_false_test_data
                                count_NIO_false_test_data_best=count_NIO_false_test_data
                                count_NIO_true_test_data_best=count_NIO_true_test_data
                                count_NIO_test_best=count_NIO_test
                                count_IO_test_best=count_IO_test
                                count_NIO_train_best=count_NIO_train
                                count_IO_train_best=count_IO_train
                                balanced_acc_test_best=balanced_acc_test
                                acc_test_best=acc_test
                                balanced_acc_all_best=balanced_acc_all
                                acc_all_best=acc_all
                                precision_IO_all_best=precision_IO_all
                                recall_IO_all_best=recall_IO_all
                                fscore_IO_all_best=fscore_IO_all
                                precision_NIO_all_best=precision_NIO_all
                                recall_NIO_all_best=recall_NIO_all
                                fscore_NIO_all_best=fscore_NIO_all
                                precision_IO_test_best=precision_IO_test
                                recall_IO_test_best=recall_IO_test
                                fscore_IO_test_best=fscore_IO_test
                                precision_NIO_test_best=precision_NIO_test
                                recall_NIO_test_best=recall_NIO_test
                                fscore_NIO_test_best=fscore_NIO_test
                                modeltrainingloss_best=modeltrainingloss
                                modelvalidationloss_best=modelvalidationloss
                                if validation_trigger==True:
                                    balanced_acc_val_best=balanced_acc_val
                                    acc_val_best=acc_val
                                    count_IO_true_val_data_best=count_IO_true_val_data
                                    count_IO_false_val_data_best=count_IO_false_val_data
                                    count_NIO_false_val_data_best=count_NIO_false_val_data
                                    count_NIO_true_val_data_best=count_NIO_true_val_data
                                    count_NIO_val_best=count_NIO_val
                                    count_IO_val_best=count_IO_val
                                    precision_IO_val_best=precision_IO_val
                                    recall_IO_val_best=recall_IO_val
                                    fscore_IO_val_best=fscore_IO_val
                                    precision_NIO_val_best=precision_NIO_val
                                    recall_NIO_val_best=recall_NIO_val
                                    fscore_NIO_val_best=fscore_NIO_val


                        count_IO_true_all_data=np.mean(count_IO_true_all_data_rand_list)
                        count_IO_false_all_data=np.mean(count_IO_false_all_data_rand_list)
                        count_NIO_false_all_data=np.mean(count_NIO_false_all_data_rand_list)
                        count_NIO_true_all_data=np.mean(count_NIO_true_all_data_rand_list)
                        count_IO_true_test_data=np.mean(count_IO_true_test_data_rand_list)
                        count_IO_false_test_data=np.mean(count_IO_false_test_data_rand_list)
                        count_NIO_false_test_data=np.mean(count_NIO_false_test_data_rand_list)
                        count_NIO_true_test_data=np.mean(count_NIO_true_test_data_rand_list)
                        count_NIO_test=np.mean(count_NIO_test_rand_list)
                        count_IO_test=np.mean(count_IO_test_rand_list)
                        count_NIO_train=np.mean(count_NIO_train_rand_list)
                        count_IO_train=np.mean(count_IO_train_rand_list)
                        balanced_acc_test=np.mean(balanced_acc_test_rand_list)
                        acc_test=np.mean(acc_test_rand_list)
                        balanced_acc_all=np.mean(balanced_acc_all_rand_list)
                        acc_all=np.mean(acc_all_rand_list)
                        precision_IO_all=np.mean(precision_IO_all_rand_list)
                        recall_IO_all=np.mean(recall_IO_all_rand_list)
                        fscore_IO_all=np.mean(fscore_IO_all_rand_list)
                        precision_NIO_all=np.mean(precision_NIO_all_rand_list)
                        recall_NIO_all=np.mean(recall_NIO_all_rand_list)
                        fscore_NIO_all=np.mean(fscore_NIO_all_rand_list)
                        precision_IO_test=np.mean(precision_IO_test_rand_list)
                        recall_IO_test=np.mean(recall_IO_test_rand_list)
                        fscore_IO_test=np.mean(fscore_IO_test_rand_list)
                        precision_NIO_test=np.mean(precision_NIO_test_rand_list)
                        recall_NIO_test=np.mean(recall_NIO_test_rand_list)
                        fscore_NIO_test=np.mean(fscore_NIO_test_rand_list)
                        modeltrainingloss=np.mean(modeltrainingloss_rand_list)
                        modelvalidationloss=np.mean(modelvalidationloss_rand_list)
                        balanced_acc_val=np.mean(balanced_acc_val_rand_list)
                        acc_val=np.mean(acc_val_rand_list)
                        count_IO_true_val_data=np.mean(count_IO_true_val_data_rand_list)
                        count_IO_false_val_data=np.mean(count_IO_false_val_data_rand_list)
                        count_NIO_false_val_data=np.mean(count_NIO_false_val_data_rand_list)
                        count_NIO_true_val_data=np.mean(count_NIO_true_val_data_rand_list)
                        count_NIO_val=np.mean(count_NIO_val_rand_list)
                        count_IO_val=np.mean(count_IO_val_rand_list)
                        precision_IO_val=np.mean(precision_IO_val_rand_list)
                        recall_IO_val=np.mean(recall_IO_val_rand_list)
                        fscore_IO_val=np.mean(fscore_IO_val_rand_list)
                        precision_NIO_val=np.mean(precision_NIO_val_rand_list)
                        recall_NIO_val=np.mean(recall_NIO_val_rand_list)
                        fscore_NIO_val=np.mean(fscore_NIO_val_rand_list)

                        print(shap_values.shape)
                        shap_values_array=np.stack(shap_values_list,axis=0)
                        print(shap_values_array.shape)
                        expected_value_array=np.stack(expected_value_list,axis=0)
                        shap_values=np.mean(shap_values_list,axis=0)
                        print(shap_values.shape)
                        expected_value=np.mean(expected_value_list,axis=0)

                        scores=[
                            ["XGB"]+[runname]+[dataname]+[xgbfraction]+[errorname]+[balancing]+[scalername]+[randomseeds_str]+
                            
                            [count_features]+[count_rows_all]+
                            [count_rows_balanced]+[count_rows_balanced_IO]+[count_rows_balanced_NIO]+
                            
                            [count_IO_test]+[count_NIO_test]+
                            [balanced_acc_test]+[precision_NIO_test]+[recall_NIO_test]+[fscore_NIO_test]+[precision_IO_test]+[recall_IO_test]+[fscore_IO_test]+[acc_test]+
                            [count_IO_test_best]+[count_NIO_test_best]+
                            [balanced_acc_test_best]+[precision_NIO_test_best]+[recall_NIO_test_best]+[fscore_NIO_test_best]+[precision_IO_test_best]+[recall_IO_test_best]+[fscore_IO_test_best]+[acc_test_best]+                           

                            [count_rows_all_IO]+[count_rows_all_NIO]+
                            [balanced_acc_all]+[precision_NIO_all]+[recall_NIO_all]+[fscore_NIO_all]+[precision_IO_all]+[recall_IO_all]+[fscore_IO_all]+[acc_all]+
                            [balanced_acc_all_best]+[precision_NIO_all_best]+[recall_NIO_all_best]+[fscore_NIO_all_best]+[precision_IO_all_best]+[recall_IO_all_best]+[fscore_IO_all_best]+[acc_all_best]+    

                            [count_IO_val]+[count_NIO_val]+
                            [balanced_acc_val]+[precision_NIO_val]+[recall_NIO_val]+[fscore_NIO_val]+[precision_IO_val]+[recall_IO_val]+[fscore_IO_val]+[acc_val]+
                            [balanced_acc_val_best]+[precision_NIO_val_best]+[recall_NIO_val_best]+[fscore_NIO_val_best]+[precision_IO_val_best]+[recall_IO_val_best]+[fscore_IO_val_best]+[acc_val_best]+
                        
                            [count_IO_true_test_data]+[count_IO_false_test_data]+[count_NIO_false_test_data]+[count_NIO_true_test_data]+
                            [count_IO_true_test_data_best]+[count_IO_false_test_data_best]+[count_NIO_false_test_data_best]+[count_NIO_true_test_data_best]+ 
                            [count_IO_true_val_data]+[count_IO_false_val_data]+[count_NIO_false_val_data]+[count_NIO_true_val_data]+
                            [count_IO_true_val_data_best]+[count_IO_false_val_data_best]+[count_NIO_false_val_data_best]+[count_NIO_true_val_data_best]+
                            [count_IO_true_all_data]+[count_IO_false_all_data]+[count_NIO_false_all_data]+[count_NIO_true_all_data]+
                            [count_IO_true_all_data_best]+[count_IO_false_all_data_best]+[count_NIO_false_all_data_best]+[count_NIO_true_all_data_best]+
                            
                            [count_IO_train]+[count_NIO_train]+[count_IO_train_best]+[count_NIO_train_best]+

                            [modeltrainingloss]+[modelvalidationloss]+
                            [modeltrainingloss_best]+[modelvalidationloss_best]+          

                            ["none"]+["none"]+["none"]+["none"]+["none"]+["none"]+["none"]+["none"]+["none"]+["none"]+["none"]+["none"]+["none"]+["none"]+["none"]+["none"]+["none"]+["none"]+["none"]+["none"]+
                            [XGB_objective_i]+[XGB_n_estimators_i]+[XGB_parallel_tree_i]+[XGB_maxdepth_i]+[XGB_eta_i]+[XGB_earlystoppingrounds_i]+[XGB_minchild_weight_i]+[XGB_max_delta_step_i]+
                            [XGB_subsample_i]+[XGB_colsample_bytree_i]+[XGB_colsample_bylevel_i]+[XGB_colsample_bynode_i]+[XGB_alpha_i]+[XGB_lambda_i]+[XGB_gamma_i]
                        ]
                        if best_export_trigger == True:
                            entryname=str(dataname)+"_"+str(xgbfraction).zfill(3)+"_"+errorname+"_XGB"
                            if balanced_acc_test > best_dict[entryname]:
                                best_dict[entryname]=balanced_acc_test.copy()
                                best_score_dict[entryname+"_scorelist"]=scores[0].copy()
                                exportdata=pd.DataFrame.from_dict(best_score_dict,orient='index',columns=columnnames)
                                with open(savepath+r"\\best_scores.csv", mode='w', newline='\n') as f:
                                    exportdata.to_csv(f, float_format='%.6f',index=False) 
                                best_trigger=True                  
                        str_time=datetime.now().strftime('%Y%m%d-%H%M%S') 
                        if XGB_modelsave_trigger == True:
                            with open(savepath+r"\\"+str_time+"_XGB_model_"+runname+"_"+dataname+"_"+errorname+"_"+str(XGB_n_estimators_i)+"_"+str(XGB_parallel_tree_i)+"_"+str(XGB_maxdepth_i)+"_"+str(XGB_eta_i), mode='wb') as f:
                                joblib.dump(XGBmodel, f)
                        globalscores=globalscores+scores

                        losscurve_length=len(losscurve_list[0])
                        for losscurve_i in losscurve_list:
                            losscurve_length=min(losscurve_length,len(losscurve_i))
                        losscurve_list_new=[]
                        for losscurve_i in losscurve_list:
                            losscurve_list_new=losscurve_list_new+[losscurve_i[:losscurve_length]]
                        losscurve_array=np.stack(losscurve_list_new,axis=0)
                        losscurve=np.mean(losscurve_array,axis=0)

                        losscurve_val_length=len(losscurve_val_list[0])
                        for losscurve_i in losscurve_val_list:
                            losscurve_val_length=min(losscurve_val_length,len(losscurve_i))
                        losscurve_val_list_new=[]
                        for losscurve_i in losscurve_val_list:
                            losscurve_val_list_new=losscurve_val_list_new+[losscurve_i[:losscurve_val_length]]
                        losscurve_val_array=np.stack(losscurve_val_list_new,axis=0)
                        losscurve_val=np.mean(losscurve_val_array,axis=0)
                        pl.figure(figsize=(1, 1), dpi=80)
                        pl.plot(losscurve, label='train')
                        pl.plot(losscurve_val, label='test')
                        pl.xlabel('iteration')
                        pl.ylabel('loss')
                        pl.title("losscurves_avg_"+entryname)
                        pl.legend()
                        fig = pl.gcf()
                        fig.set_size_inches(18.5, 10.5)
                        fig.tight_layout()
                        str_time=datetime.now().strftime('%Y%m%d-%H%M%S')
                        if best_trigger == True:
                            with open(savepath+r"\\"+entryname+"_avg_losscurve.eps", mode='w') as f:
                                pl.savefig(f,bbox_inches='tight', format='eps')
                            with open(savepath+r"\\"+entryname+"_avg_losscurve.png", mode='wb') as f:
                                pl.savefig(f,bbox_inches='tight', format='png')   
                        pl.close(True)


                        pl.figure(figsize=(1, 1), dpi=80)
                        pl.plot(XGBlossplotdata['validation_0']['logloss'], label='train')
                        pl.plot(XGBlossplotdata['validation_1']['logloss'], label='test')
                        pl.xlabel('iteration')
                        pl.ylabel('loss')
                        pl.title("losscurves_best_"+entryname)
                        pl.legend()
                        fig = pl.gcf()
                        fig.set_size_inches(18.5, 10.5)
                        fig.tight_layout()
                        str_time=datetime.now().strftime('%Y%m%d-%H%M%S')
                        if best_trigger == True:
                            with open(savepath+r"\\"+entryname+"_best_losscurve.eps", mode='w') as f:
                                pl.savefig(f,bbox_inches='tight', format='eps')
                            with open(savepath+r"\\"+entryname+"_best_losscurve.png", mode='wb') as f:
                                pl.savefig(f,bbox_inches='tight', format='png')   
                            with open(savepath+r"\\"+entryname+"_bestmodel", mode='wb') as f:
                                joblib.dump(XGBmodel, f)    
                        pl.close(True)
                        shap_values_abs=np.sum(abs(shap_values),axis=0)
                        xgbidentifier=str("XGB_"+runname+"_"+dataname+"_"+balancing+"_"+errorname+"_"+scalername)
                        pd_dic={"xgbidentifier":xgbidentifier,"importance gain" :XGBmodel.feature_importances_,"shap value":shap_values_abs}
                        pd_dataset=pd.DataFrame(pd_dic)        
                        str_time=datetime.now().strftime('%Y%m%d-%H%M%S')
                        if importancesave_trigger==True:
                            with open(savepath+r"\\"+str_time+"_importance_XGB_"+runname+"_"+dataname+"_"+errorname+".csv", mode='w', newline='\n') as f:
                                pd_dataset.to_csv(f, float_format='%.6f',index=True) 

                        if ((SHAP_trigger ==True) or (best_trigger==True)):

                            #SHAP PLOT
                            pl.figure(figsize=(8.3,11.7))
                            pl.title("SHAP "+entryname[:-4]+", datapoints: "+str(X_all.shape[0]))
                            shap.summary_plot(shap_values,pd.DataFrame(columns=X_all.columns,data=X_all.values),max_display=30,show=False)
                            fig = pl.gcf()
                            fig.set_size_inches(8.3, 11.7)
                            #fig.tight_layout()
                            str_time=datetime.now().strftime('%Y%m%d-%H%M%S')
                            if SHAP_trigger == True:
                                with open(savepath+r"\\"+str_time+"_SHAPplot_XGB_"+runname+"_"+dataname+"_"+errorname+".eps", mode='w') as f:
                                    pl.savefig(f,bbox_inches='tight', format='eps')
                                with open(savepath+r"\\"+str_time+"_SHAPplot_XGB_"+runname+"_"+dataname+"_"+errorname+".png", mode='wb') as f:
                                    pl.savefig(f,bbox_inches='tight', format='png')
                            if best_trigger == True:
                                with open(savepath+r"\\"+entryname+"_avg_SHAP_plot.png", mode='wb') as f:
                                    pl.savefig(f,bbox_inches='tight', format='png')
                                with open(savepath+r"\\"+entryname+"_avg_SHAP_plot.eps", mode='w') as f:
                                    pl.savefig(f,bbox_inches='tight', format='eps')                            
                            pl.close(True)

                            #SHAP DECISION PLOT
                            pl.figure(figsize=(8.3,11.7))
                            pl.title("SHAP decision plot "+entryname[:-4]+", datapoints: "+str(X_all.shape[0]))
                            #s=list(np.argsort(-(abs(shap_values)).mean(0)))
                            #shap_data=X_pd.iloc[:,s]
                            #shap_values = explainer.shap_values(shap_data)
                            
                            
                            shapdpsubset=list(random.sample(range(len(shap_values)), 20))
                            shap.decision_plot(expected_value,shap_values[shapdpsubset],X_all.iloc[shapdpsubset],show=False,feature_order="importance",feature_display_range=slice(-1, -31, -1))
                            fig = pl.gcf()
                            fig.set_size_inches(8.3, 11.7)            
                            str_time=datetime.now().strftime('%Y%m%d-%H%M%S')
                            if SHAP_trigger == True:
                                with open(savepath+r"\\"+str_time+"_SHAPdecisionplot_XGB_"+runname+"_"+dataname+"_"+errorname+".eps", mode='w') as f:
                                    pl.savefig(f,bbox_inches='tight', format='eps')
                                with open(savepath+r"\\"+str_time+"_SHAPdecisionplot_XGB_"+runname+"_"+dataname+"_"+errorname+".png", mode='wb') as f:
                                    pl.savefig(f,bbox_inches='tight', format='png')
                            if best_trigger == True:                          
                                with open(savepath+r"\\"+entryname+"_avg_SHAP_decision_plot.png", mode='wb') as f:
                                    pl.savefig(f,bbox_inches='tight', format='png')
                                with open(savepath+r"\\"+entryname+"_avg_SHAP_decision_plot.eps", mode='w') as f:
                                    pl.savefig(f,bbox_inches='tight', format='eps')                            
                            pl.close(True)
                        pd_dataset=pd.DataFrame(columns=columnnames,data=globalscores)
                        str_time=datetime.now().strftime('%Y%m%d-%H%M%S')
                        with open(savepath+r"\\scores_saveslot_"+str(saveslotcounter)+".csv", mode='w', newline='\n') as f:
                            pd_dataset.to_csv(f, float_format='%.6f',index=True) 
                        saveslotcounter=saveslotcounter+1
                        if saveslotcounter>10:
                            saveslotcounter=1                    
                        print("Ende XGBoost")


if ((XGB_trigger == True) & (NN_trigger == False)):
    pd_dataset=pd.DataFrame(columns=columnnames,data=globalscores)
    str_time=datetime.now().strftime('%Y%m%d-%H%M%S')
    with open(savepath+r"\\"+str_time+"_scores_"+runname+"_"+folderexplanation+".csv", mode='w', newline='\n') as f:
        pd_dataset.to_csv(f, float_format='%.6f',index=True) 
    "Ende, XGB Modell"
elif ((XGB_trigger == False) & (NN_trigger == True)):
    pd_dataset=pd.DataFrame(columns=columnnames,data=globalscores)
    str_time=datetime.now().strftime('%Y%m%d-%H%M%S')
    with open(savepath+r"\\"+str_time+"_scores_"+runname+"_"+folderexplanation+".csv", mode='w', newline='\n') as f:
        pd_dataset.to_csv(f, float_format='%.6f',index=True) 
    "Ende, NN Modell"
elif ((XGB_trigger == True) & (NN_trigger == True)):
    pd_dataset=pd.DataFrame(columns=columnnames,data=globalscores)
    str_time=datetime.now().strftime('%Y%m%d-%H%M%S')   
    with open(savepath+r"\\"+str_time+"_scores_"+runname+"_"+folderexplanation+".csv", mode='w', newline='\n') as f:
        pd_dataset.to_csv(f, float_format='%.6f',index=True)
    "Ende, XGB und NN Modell"
else:
    "Ende, kein Modell"


#NN_modelscores_column_names=["name","training loss","validation loss"]
#pd_dataset=pd.DataFrame(columns=NN_modelscores_column_names,data=NN_modelscores_list)
#str_time=datetime.now().strftime('%Y%m%d-%H%M%S')
#with open(savepath+r"\\"+str_time+"_Modelscores.csv", mode='w', newline='\n') as f:
#    pd_dataset.to_csv(f, float_format='%.6f',index=True)      
        
'''
data=pd.DataFrame(joblib.load(path+"/all_but_labels_20190215"))
print(data.shape)
with open(path+"/all_but_labels_20190215.csv", mode='w', newline='\n') as f:
    data.to_csv(f, float_format='%.6f',index=True,lineterminator='\r\n')
with open(path+"/all_but_labels_20190215_first.csv", mode='w', newline='\n') as f:
    data.iloc[0].to_csv(f, float_format='%.6f',index=True,lineterminator='\r\n')
with open(path+"all_but_labels_20190215.pickle",'wb') as f:
    data.to_pickle(f) 

data=pd.DataFrame(joblib.load(path+"/all_Features_20190215"))
print(data.shape)
with open(path+"/all_features_20190215.csv", mode='w', newline='\n') as f:
    data.to_csv(f, float_format='%.6f',index=True,lineterminator='\r\n')
with open(path+"/all_features_20190215_first.csv", mode='w', newline='\n') as f:
    data.iloc[0].to_csv(f, float_format='%.6f',index=True,lineterminator='\r\n')
with open(path+"/all_features_20190215.pickle",'wb') as f:
    data.to_pickle(f) 


data=pd.DataFrame(joblib.load(path+"/all_Features_cat_expanded_20190215"))
print(data.shape)
with open(path+"/all_Features_cat_expanded_20190215.csv", mode='w', newline='\n') as f:
    data.to_csv(f, float_format='%.6f',index=True,lineterminator='\r\n') 
with open(path+"/all_Features_cat_expanded_20190215_first.csv", mode='w', newline='\n') as f:
    data.iloc[0].to_csv(f, float_format='%.6f',index=True,lineterminator='\r\n')
with open(path+"/all_Features_cat_expanded_20190215.pickle",'wb') as f:
    data.to_pickle(f) 


data=pd.DataFrame(joblib.load(path+"/all_Features_cat_expanded_20190313"))
print(data.shape)
with open(path+"/all_Features_cat_expanded_20190313.csv", mode='w', newline='\n') as f:
    data.to_csv(f, float_format='%.6f',index=True,lineterminator='\r\n') 
with open(path+"/all_Features_cat_expanded_20190313_first.csv", mode='w', newline='\n') as f:
    data.iloc[0].to_csv(f, float_format='%.6f',index=True,lineterminator='\r\n')
with open(path+"/all_Features_cat_expanded_20190313.pickle",'wb') as f:
    data.to_pickle(f) 

data=pd.DataFrame(joblib.load(path+"/Skalare_complete_incl_Features_20190319"))
with open(path+"/Skalare_complete_incl_Features_20190319.csv", mode='wb') as f:
    data.to_csv(f, float_format='%.6f',index=True,lineterminator='\r\n') 
with open(path+"/Skalare_complete_incl_Features_20190319_first.csv", mode='wb') as f:
    data.iloc[0].to_csv(f, float_format='%.6f',index=True,lineterminator='\r\n')
with open(path+"/Skalare_complete_incl_Features_20190319.pickle",'wb') as f:
    data.to_pickle(f) 
print(data.shape)


data=pd.DataFrame(joblib.load(path+"/all_Labels"))
with open(path+"/all_Labels.csv", mode='wb') as f:
    data.to_csv(f, float_format='%.6f',index=True,lineterminator='\r\n') 
with open(path+"/all_Labels_first.csv", mode='wb') as f:
    data.iloc[0].to_csv(f, float_format='%.6f',index=True,lineterminator='\r\n')
with open(path+"/all_Labels.pickle",'wb') as f:
    data.to_pickle(f) 
print(data.shape)

data=pd.DataFrame(joblib.load(path+"/all_Labels_20190214"))
with open(path+"/all_Labels_20190214.csv", mode='wb') as f:
    data.to_csv(f, float_format='%.6f',index=True,lineterminator='\r\n') 
with open(path+"/all_Labels_20190214_first.csv", mode='wb') as f:
    data.iloc[0].to_csv(f, float_format='%.6f',index=True,lineterminator='\r\n')
with open(path+"/all_Labels_20190214.pickle",'wb') as f:
    data.to_pickle(f) 
print(data.shape)

data=pd.DataFrame(joblib.load(path+"/all_Labels_20190215"))
with open(path+"/all_Labels_20190215.csv", mode='wb') as f:
    data.to_csv(f, float_format='%.6f',index=True,lineterminator='\r\n') 
with open(path+"/all_Labels_20190215_first.csv", mode='wb') as f:
    data.iloc[0].to_csv(f, float_format='%.6f',index=True,lineterminator='\r\n')
with open(path+"/all_Labels_20190215.pickle",'wb') as f:
    data.to_pickle(f) 
print(data.shape)
'''