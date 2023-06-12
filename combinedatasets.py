
targetcolumns=["FEHLERART_BEZ"]  #DMCODE,FEHLERART_BEZ,FEHLERKLAsE,IO/NIO,FEHLERCLUSTER,Fehlerort_BMW
targetnumber=1
targeterrors=[["Gasblase","Kaltlauf","undicht","Poros"],"Gasblase","Kaltlauf","undicht","Poros"] #targeterrors=["Gasblase","Kaltlauf","Kernfehler","Lunker","undicht","Poros","Prozessfehler","Sonstiges"]


with open(path+"/all_Labels_20190215_ohne_Umlaute_gruppiert_iofilled.csv") as fh:
    orgdata=pd.read_csv(fh)
    orgdata.drop(orgdata.columns[0],axis=1,inplace=True)
#print(orgdata)

y1=orgdata[["DMCODE","FEHLERART_BEZ"]].copy()
print(y1)
with open(path+"/all_Features_cat_expanded_20190313_modified_names.csv") as fh:
    orgdata=pd.read_csv(fh)
print(orgdata)
for i in timedelta_columns:
    orgdata[i] = pd.to_timedelta(orgdata[i]).astype('timedelta64[s]').astype(int)
totaldata1=pd.merge(y1,orgdata, on='DMCODE')
print(totaldata1)

with open(path+"/20230213-184900_tsdata.csv") as fh:
    tsdata=pd.read_csv(fh)
    tsdata.drop(tsdata.columns[0],axis=1,inplace=True)
print(tsdata)

totaldata=pd.merge(totaldata1,tsdata,on="DMCODE")
print(totaldata)

with open(path+"/totaldata_13022023.csv", mode='wb') as f:
    totaldata.to_csv(f, float_format='%.6f',index=True,lineterminator='\r\n') 
with open(path+"/totaldata_13022023_first.csv", mode='wb') as f:
    totaldata.iloc[0].to_csv(f, float_format='%.6f',index=True,lineterminator='\r\n')
with open(path+"/totaldata_13022023.pickle",'wb') as f:
    totaldata.to_pickle(f) 
