import numpy as np
import pandas as pd
import joblib
from sklearn.feature_selection import SelectFromModel
import os

'v2: Verwendung von newnames zur Verschleierung'

def prepare_datasets(orgdata,triggerlist,xgbfraction,XGBfiltermodel_base,XGBfiltermodel_fe,XGBfiltermodel_tsfresh):

    base_trigger,base_fe_trigger,base_tsfresh_trigger,kok_base_trigger,kok_fe_trigger,kok_tsfresh_trigger,nest_base_trigger,nest_fe_trigger,nest_tsfresh_trigger = triggerlist
    base_trigger=triggerlist[0]
    base_fe_trigger=triggerlist[1]
    base_tsfresh_trigger=triggerlist[2]
    kok_base_trigger=triggerlist[3]
    kok_fe_trigger=triggerlist[4]
    kok_tsfresh_trigger=triggerlist[5]
    nest_base_trigger=triggerlist[6]
    nest_fe_trigger=triggerlist[7]
    nest_tsfresh_trigger=triggerlist[8]


   
    datalist=[]
    datanamelist=[]
    xgbfraction_datalist=[]

    zeitschritte=list(range(0,32,1))
    ts_list=["KGA","TE1","TE2","TE3","TE4","TE6","TE7",
        "Wms1","Wms2","Wms3","Wms4","Wms6","Wms7",
        "WTE1","WTE2","WTE3","WTE5","WTE7","WTE8"]
    tscolumnnames=[]
    for i in ts_list:
        for j in zeitschritte:
            tsstr=i+"_"+str(j)
            tscolumnnames=tscolumnnames+[tsstr]
    errorcolumnname=["FEHLERART_BEZ"]
    scalar_columnnames=["O_T","LV_T","LV_P","WV_T","WV_P","O_FH","GP_I_VFt","GP_I_Gt","GP_I_Gp","trigger_p","t_offen",
                    "KET1","KET2","KET3","KET5","KET7","KET8","GP_Io_1","GP_Io_2","GP_Io_3","GP_Io_o4",
                    "GP_Io_5","GP_Iv_1","GP_Iv_2","GP_Iv_3","GP_Iv_4","GP_Iv_5","GP_Lo_1",
                    "GP_Lo_2","GP_Lo_3","GP_Lo_4","GP_Lo_5","T_env","Feuchte_env","GP_Gt","IDG1_0","IDG1_1",
                    "IDG2_f00","IDG2_f01","IDG2_o00","IDG2_o99","IDG2_s00","IDG2_s99","IDG2_x00","IDG2_x01","IDG3_68","IDG4_69","IDG5_5","IDG5_6",
                    "IDG5_7","IDG6_2","IDG6_4","IDG6_5","IDG6_6","IDG6_7","IDG6_8","IDG6_9","IDG6_10",
                    "IDG6_11","IDG6_12","IDG6_13","IDG6_14","KOKNR_1","KOKNR_2","KOKNR_3","KOKNR_4","NEST_1","NEST_2"]
    
    base_columnnames=scalar_columnnames+tscolumnnames+errorcolumnname

    fe_columnnames=["FE_t_KE_Ab","FE_n_BT_kok_prep","FE_n_BT_I_prep","FE_n_BT_kok_ges","FE_n_BT_I_ges",
                    "FE_n_BT_start","FE_t_start","FE_t_start_S","FE_n_BT_start_S","FE_O_T_d","FE_I_GR_d",
                    "FE_C_TE1_2","FE_C_TE1_3","FE_C_TE1_5","FE_C_TE1_7","FE_C_TE1_8","FE_C_TE2_3","FE_C_TE2_5","FE_C_TE2_7",
                    "FE_C_TE2_8","FE_C_TE3_5","FE_C_TE3_7","FE_C_TE3_8","FE_C_TE5_7","FE_C_TE5_8","FE_C_TE7_8","FE_TE1_delta",
                    "FE_TE2_delta","FE_TE3_delta","FE_TE5_delta","FE_TE7_delta","FE_TE8_delta"]
    base_fe_columnnames=scalar_columnnames+fe_columnnames+tscolumnnames+errorcolumnname

    tsfresh_columnnames=list(orgdata.loc[:,"TE1_tsf1":"TE8_tsf14"].columns)
    base_tsfresh_columnnames=scalar_columnnames+fe_columnnames+tscolumnnames+tsfresh_columnnames+errorcolumnname

    # Basisscore alle Rohdaten ohne TSfresh
    base_data = orgdata[base_columnnames]
    base_fe_data = orgdata[base_fe_columnnames]
    base_tsfresh_data = orgdata[base_tsfresh_columnnames]
  
    kok_droplist=["KOKNR_1","KOKNR_2","KOKNR_3","KOKNR_4"]
    nest_droplist=["NEST_1","NEST_2"]
    he_droplist=kok_droplist+nest_droplist


    base_droplist=[]
    fe_droplist=[]
    tsfresh_droplist=[]
    if ((True in [base_trigger,kok_base_trigger,nest_base_trigger]) and (xgbfraction != 100)):
        selectmodel = SelectFromModel(XGBfiltermodel_base, prefit=True,threshold=-np.inf,max_features=int(base_data.shape[1]*xgbfraction/100))
        xgb_base_data=base_data.drop("FEHLERART_BEZ",axis=1)
        xgbcolumnnames=selectmodel.get_feature_names_out(xgb_base_data.columns)
        for i in he_droplist:
            if not(i in xgbcolumnnames):
                base_droplist=base_droplist+[i]
        xgb_base_data = pd.DataFrame(columns=xgbcolumnnames,data=selectmodel.transform(xgb_base_data))
        xgb_base_data["FEHLERART_BEZ"] = base_data["FEHLERART_BEZ"]
        xgb_base_data["KOKNR_1"] = base_data["KOKNR_1"]
        xgb_base_data["KOKNR_2"] = base_data["KOKNR_2"]
        xgb_base_data["KOKNR_3"] = base_data["KOKNR_3"]
        xgb_base_data["KOKNR_4"] = base_data["KOKNR_4"]
        xgb_base_data["NEST_1"] = base_data["NEST_1"]
        xgb_base_data["NEST_2"] = base_data["NEST_2"]
    else:
        xgb_base_data=base_data
    if ((True in [base_fe_trigger,kok_fe_trigger,nest_fe_trigger]) and (xgbfraction != 100)):
        selectmodel = SelectFromModel(XGBfiltermodel_fe, prefit=True,threshold=-np.inf,max_features=int(base_fe_data.shape[1]*xgbfraction/100))
        xgb_base_fe_data = base_fe_data.drop("FEHLERART_BEZ",axis=1)
        xgbcolumnnames=selectmodel.get_feature_names_out(xgb_base_fe_data.columns)  
        for i in he_droplist:
            if not(i in xgbcolumnnames):
                fe_droplist=fe_droplist+[i]        
        xgb_base_fe_data = pd.DataFrame(columns=xgbcolumnnames,data=selectmodel.transform(xgb_base_fe_data))
        xgb_base_fe_data["FEHLERART_BEZ"] = base_fe_data["FEHLERART_BEZ"]
        xgb_base_fe_data["KOKNR_1"] = base_fe_data["KOKNR_1"]
        xgb_base_fe_data["KOKNR_2"] = base_fe_data["KOKNR_2"]
        xgb_base_fe_data["KOKNR_3"] = base_fe_data["KOKNR_3"]
        xgb_base_fe_data["KOKNR_4"] = base_fe_data["KOKNR_4"]
        xgb_base_fe_data["NEST_1"] = base_fe_data["NEST_1"]
        xgb_base_fe_data["NEST_2"] = base_fe_data["NEST_2"]
    else:
        xgb_base_fe_data=base_fe_data

    if ((True in [base_tsfresh_trigger,kok_tsfresh_trigger,nest_tsfresh_trigger]) and (xgbfraction != 100)):
        selectmodel = SelectFromModel(XGBfiltermodel_tsfresh, prefit=True,threshold=-np.inf,max_features=int(base_tsfresh_data.shape[1]*xgbfraction/100))
        xgb_base_tsfresh_data = base_tsfresh_data.drop("FEHLERART_BEZ",axis=1)
        xgbcolumnnames=selectmodel.get_feature_names_out(xgb_base_tsfresh_data.columns)
        for i in he_droplist:
            if not(i in xgbcolumnnames):
                tsfresh_droplist=tsfresh_droplist+[i]        
        xgb_base_tsfresh_data = pd.DataFrame(columns=xgbcolumnnames,data=selectmodel.transform(xgb_base_tsfresh_data))
        xgb_base_tsfresh_data["FEHLERART_BEZ"] = base_tsfresh_data["FEHLERART_BEZ"]
        xgb_base_tsfresh_data["KOKNR_1"] = base_tsfresh_data["KOKNR_1"]
        xgb_base_tsfresh_data["KOKNR_2"] = base_tsfresh_data["KOKNR_2"]
        xgb_base_tsfresh_data["KOKNR_3"] = base_tsfresh_data["KOKNR_3"]
        xgb_base_tsfresh_data["KOKNR_4"] = base_tsfresh_data["KOKNR_4"]
        xgb_base_tsfresh_data["NEST_1"] = base_tsfresh_data["NEST_1"]
        xgb_base_tsfresh_data["NEST_2"] = base_tsfresh_data["NEST_2"]
    else:
        xgb_base_tsfresh_data=base_tsfresh_data            

    if base_trigger == True:
        datalist=datalist+[xgb_base_data.drop(base_droplist,axis=1).copy()]
        datanamelist=datanamelist+["base_data"]
        xgbfraction_datalist=xgbfraction_datalist+[xgbfraction]

    if base_fe_trigger == True:
        datalist=datalist+[xgb_base_fe_data.drop(fe_droplist,axis=1).copy()]
        datanamelist=datanamelist+["fe_data"]
        xgbfraction_datalist=xgbfraction_datalist+[xgbfraction]

    if base_tsfresh_trigger == True:
        datalist=datalist+[xgb_base_tsfresh_data.drop(tsfresh_droplist,axis=1).copy()]
        datanamelist=datanamelist+["tsfresh_data"]
        xgbfraction_datalist=xgbfraction_datalist+[xgbfraction]

    kok_base_droplist=base_droplist.copy()
    kok_fe_droplist=fe_droplist.copy()
    kok_tsfresh_droplist=tsfresh_droplist.copy()
    for i in kok_droplist:
        if not(i in kok_base_droplist):
            kok_base_droplist=kok_base_droplist+[i]
        if not(i in kok_fe_droplist):
            kok_fe_droplist=kok_fe_droplist+[i]
        if not(i in kok_tsfresh_droplist):
            kok_tsfresh_droplist=kok_tsfresh_droplist+[i]               
    if kok_base_trigger == True:
        kok1_xgb_base_data = xgb_base_data[xgb_base_data["KOKNR_1"]== 1]
        datalist=datalist+[kok1_xgb_base_data.drop(kok_base_droplist,axis=1).copy()]
        datanamelist=datanamelist+["kok1_base_data"]
        xgbfraction_datalist=xgbfraction_datalist+[xgbfraction]
        kok2_xgb_base_data = xgb_base_data[xgb_base_data["KOKNR_2"]== 1]
        datalist=datalist+[kok2_xgb_base_data.drop(kok_base_droplist,axis=1).copy()]
        datanamelist=datanamelist+["kok2_base_data"]
        xgbfraction_datalist=xgbfraction_datalist+[xgbfraction]
        kok3_xgb_base_data = xgb_base_data[xgb_base_data["KOKNR_3"]== 1]
        datalist=datalist+[kok3_xgb_base_data.drop(kok_base_droplist,axis=1).copy()]
        datanamelist=datanamelist+["kok3_base_data"]
        xgbfraction_datalist=xgbfraction_datalist+[xgbfraction]
        kok4_xgb_base_data = xgb_base_data[xgb_base_data["KOKNR_4"]== 1]
        datalist=datalist+[kok4_xgb_base_data.drop(kok_base_droplist,axis=1).copy()]
        datanamelist=datanamelist+["kok4_base_data"]
        xgbfraction_datalist=xgbfraction_datalist+[xgbfraction]
    if kok_fe_trigger == True:
        kok1_xgb_fe_data = xgb_base_fe_data[xgb_base_fe_data["KOKNR_1"]== 1]
        datalist=datalist+[kok1_xgb_fe_data.drop(kok_fe_droplist,axis=1).copy()]
        datanamelist=datanamelist+["kok1_fe_data"]
        xgbfraction_datalist=xgbfraction_datalist+[xgbfraction]
        kok2_xgb_fe_data = xgb_base_fe_data[xgb_base_fe_data["KOKNR_2"]== 1]
        datalist=datalist+[kok2_xgb_fe_data.drop(kok_fe_droplist,axis=1).copy()]
        datanamelist=datanamelist+["kok2_fe_data"]
        xgbfraction_datalist=xgbfraction_datalist+[xgbfraction]
        kok3_xgb_fe_data = xgb_base_fe_data[xgb_base_fe_data["KOKNR_3"]== 1]
        datalist=datalist+[kok3_xgb_fe_data.drop(kok_fe_droplist,axis=1).copy()]
        datanamelist=datanamelist+["kok3_fe_data"]
        xgbfraction_datalist=xgbfraction_datalist+[xgbfraction]
        kok4_xgb_fe_data = xgb_base_fe_data[xgb_base_fe_data["KOKNR_4"]== 1]
        datalist=datalist+[kok4_xgb_fe_data.drop(kok_fe_droplist,axis=1).copy()]
        datanamelist=datanamelist+["kok4_fe_data"]
        xgbfraction_datalist=xgbfraction_datalist+[xgbfraction]
    if kok_tsfresh_trigger == True:
        kok1_xgb_tsfresh_data = xgb_base_tsfresh_data[xgb_base_tsfresh_data["KOKNR_1"]== 1]
        datalist=datalist+[kok1_xgb_tsfresh_data.drop(kok_tsfresh_droplist,axis=1).copy()]
        datanamelist=datanamelist+["kok1_tsfresh_data"]
        xgbfraction_datalist=xgbfraction_datalist+[xgbfraction]
        kok2_xgb_tsfresh_data = xgb_base_tsfresh_data[xgb_base_tsfresh_data["KOKNR_2"]== 1]
        datalist=datalist+[kok2_xgb_tsfresh_data.drop(kok_tsfresh_droplist,axis=1).copy()]
        datanamelist=datanamelist+["kok2_tsfresh_data"]
        xgbfraction_datalist=xgbfraction_datalist+[xgbfraction]
        kok3_xgb_tsfresh_data = xgb_base_tsfresh_data[xgb_base_tsfresh_data["KOKNR_3"]== 1]
        datalist=datalist+[kok3_xgb_tsfresh_data.drop(kok_tsfresh_droplist,axis=1).copy()]
        datanamelist=datanamelist+["kok3_tsfresh_data"]
        xgbfraction_datalist=xgbfraction_datalist+[xgbfraction]
        kok4_xgb_tsfresh_data = xgb_base_tsfresh_data[xgb_base_tsfresh_data["KOKNR_4"]== 1]
        datalist=datalist+[kok4_xgb_tsfresh_data.drop(kok_tsfresh_droplist,axis=1).copy()]
        datanamelist=datanamelist+["kok4_tsfresh_data"]
        xgbfraction_datalist=xgbfraction_datalist+[xgbfraction]

    nest_base_droplist=base_droplist.copy()
    nest_fe_droplist=fe_droplist.copy()
    nest_tsfresh_droplist=tsfresh_droplist.copy()
    for i in nest_droplist:
        if not(i in nest_base_droplist):
            nest_base_droplist=nest_base_droplist+[i]
        if not(i in nest_fe_droplist):
            nest_fe_droplist=nest_fe_droplist+[i]
        if not(i in nest_tsfresh_droplist):
            nest_tsfresh_droplist=nest_tsfresh_droplist+[i] 


    if nest_base_trigger == True:
        xgb_nest1_base_data = xgb_base_data[xgb_base_data["NEST_1"]== 1]
        datalist=datalist+[xgb_nest1_base_data.drop(nest_base_droplist,axis=1).copy()]
        datanamelist=datanamelist+["nest1_base_data"]
        xgbfraction_datalist=xgbfraction_datalist+[xgbfraction]
        xgb_nest2_base_data = xgb_base_data[xgb_base_data["NEST_2"]== 1]
        datalist=datalist+[xgb_nest2_base_data.drop(nest_base_droplist,axis=1).copy()]
        datanamelist=datanamelist+["nest2_base_data"]
        xgbfraction_datalist=xgbfraction_datalist+[xgbfraction]
    if nest_fe_trigger == True:
        xgb_nest1_base_fe_data = xgb_base_fe_data[xgb_base_fe_data["NEST_1"]== 1]
        datalist=datalist+[xgb_nest1_base_fe_data.drop(nest_fe_droplist,axis=1).copy()]
        datanamelist=datanamelist+["nest1_base_fe_data"]
        xgbfraction_datalist=xgbfraction_datalist+[xgbfraction]
        xgb_nest2_base_fe_data = xgb_base_fe_data[xgb_base_fe_data["NEST_2"]== 1]
        datalist=datalist+[xgb_nest2_base_fe_data.drop(nest_fe_droplist,axis=1).copy()]
        datanamelist=datanamelist+["nest2_base_fe_data"]
        xgbfraction_datalist=xgbfraction_datalist+[xgbfraction]
    if nest_tsfresh_trigger == True:
        xgb_nest1_base_tsfresh_data = xgb_base_tsfresh_data[xgb_base_tsfresh_data["NEST_1"]== 1]
        datalist=datalist+[xgb_nest1_base_tsfresh_data.drop(nest_tsfresh_droplist,axis=1).copy()]
        datanamelist=datanamelist+["nest1_base_tsfresh_data"]
        xgbfraction_datalist=xgbfraction_datalist+[xgbfraction]
        xgb_nest2_base_tsfresh_data = xgb_base_tsfresh_data[xgb_base_tsfresh_data["NEST_2"]== 1]
        datalist=datalist+[xgb_nest2_base_tsfresh_data.drop(nest_tsfresh_droplist,axis=1).copy()]
        datanamelist=datanamelist+["nest2_base_tsfresh_data"]
        xgbfraction_datalist=xgbfraction_datalist+[xgbfraction]

    return(datalist,datanamelist,xgbfraction_datalist)
#for targeterror:
    #input: triggerlist,targeterror,
    #base,fe,tsfresh,
    #nest,kok,
    #xgbfilter_base,xgbfilter_fe,xgbfraction
    #xgbfilter_nest,xgbfilter_kok
    #output: datalist,datanamelist,xgbfraction_list