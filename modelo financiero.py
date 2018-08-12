
# coding: utf-8

# In[ ]:


import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[ ]:


# Lectura y Obtención de archivos .csv
Path ='/media/anmrodriguezsa/DA4E639F4E6372E5/Users/Ángela/Desktop/Proyecto1/acciones'

allFiles = glob.glob(Path + "/*.csv")
Vt = []
list_ = []
flagFirst = 0
    
for file_ in allFiles:
    dataAccion = pd.read_csv(file_,index_col=None, header=0)  #Lectura de archivos .csv
    nf = str(file_)
    x  = dataAccion["Último"].values   #Selección de los valores del precio de cierre de las acciones
    x = [s. replace('.','') for s in x] 
    x = [s. replace(',','.') for s in x]
    v = [float(s) for s in x] 
   
    closingPrice = np.flipud(v)
    vt = closingPrice[1:] /closingPrice[0:-1] #Cálculo de vt (según artículo guía)
    
    if flagFirst==0:
        flagFirst = 1
        Vt = vt
    else:
        Vt = np.vstack((Vt,vt)) #"transpone"
        plt.plot(vt)
        plt.title(file_[95:-4] + str(len(Vt)))
        plt.show()

onesColumn = np.ones((1,Vt.shape[1]));
Vt = np.vstack((onesColumn,Vt)) # ubica elementos en la parte inferior - en este caso ubica filas-
print(Vt)

#plt.matshow(Vt)   
