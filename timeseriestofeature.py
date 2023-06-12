zeitschritte=list(range(0,160000,5000))
ts_list=["KGABS","TEMP_1","TEMP_2","TEMP_3","TEMP_4","TEMP_6","TEMP_7",
    "DURCHFL_1","DURCHFL_2","DURCHFL_3","DURCHFL_4","DURCHFL_6","DURCHFL_7",
    "THERMO_1","THERMO_2","THERMO_3","THERMO_5","THERMO_7","THERMO_8"]
tscolumnnames=["DMCODE"]
for i in ts_list:
    for j in zeitschritte:
        tsstr=i+"_"+str(j)
        tscolumnnames=tscolumnnames+[tsstr]

counter1=0
counter2=0
idata=list()
t=time.time()
for i in DMC_list:
    counter1=counter1+1
    jdata=[i]
    for j in ts_list:
        kdata=[]
        for k in zeitschritte:
            value=tsdata.loc[(tsdata["DMCODE"]==i) & (tsdata["ZEIT"]==k),j].values
            kdata=kdata+[value[0]]
        jdata=jdata+kdata
    if i == DMC_list[0]:
        idata=[jdata]
    else:
        idata=idata+[jdata]
    if counter1 == 10:
        print("abgeschlossen:",counter1+counter2)
        print("Fortschritt:",int((counter1+counter2)/220),"%")
        print("vergangene Zeit:",time.time()-t)
        counter1=0
        counter2=counter2+10
        if counter2 % 100 == 0:
            print("data saved")
            pd_dataset=pd.DataFrame(columns=tscolumnnames,data=idata)
            str_time=datetime.now().strftime('%Y%m%d-%H%M%S') 
            with open(path+r"\\"+str_time+"_tsdata.csv", mode='w', newline='\n') as f:
                pd_dataset.to_csv(f, float_format='%.6f',index=True) 
pd_dataset=pd.DataFrame(columns=tscolumnnames,data=idata)
str_time=datetime.now().strftime('%Y%m%d-%H%M%S') 
with open(path+r"\\"+str_time+"_tsdata.csv", mode='w', newline='\n') as f:
    pd_dataset.to_csv(f, float_format='%.6f',index=True) 