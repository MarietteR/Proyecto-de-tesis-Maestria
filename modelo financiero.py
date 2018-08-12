
# coding: utf-8

# In[ ]:


import pandas as pd
from pd import Series
import numpy as np
import matplotlib.pyplot as plt
import openpyxl as opxl
import os


# In[ ]:


from os import walk

for (dirpath,dirnames,filenames) in walk('acciones'):
    Vt = [];
    flagFirst = 0;
    for i in filenames:
        dataAccion = pd.read_csv('acciones/'+i);
"""
        nf = str(i);
        x  = dataAccion["Último"].values
        x = [s. replace('.','') for s in x]; 
        x = [s. replace(',','.') for s in x];
        v =  [float(s) for s in x];
        
        closingPrice = np.flipud(v);
        vt = closingPrice[1:] /closingPrice[0:-1]
        
        if flagFirst==0:
            flagFirst = 1;
            Vt = vt;
        else:

            Vt = np.vstack((Vt,vt))
            plt.plot(vt);
            plt.title(i[:-4] + str(len(Vt)))
            plt.show();        

onesColumn = np.ones((1,Vt.shape[1]));
Vt = np.vstack((onesColumn,Vt))
print(Vt)
        
Vt
plt.matshow(Vt)
"""



