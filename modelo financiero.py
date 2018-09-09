
# coding: utf-8

# In[ ]:


#Librerias
import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cvxopt import matrix, solvers
#from StringIO import StringIO

# In[ ]:


path ='/media/anmrodriguezsa/Datos/Dropbox/Universidad/Maestría en Actuaría y finanzas/Proyecto investigación Maestría/Python/acciones'
# In[ ]:


#################################################
#  Matriz de precios de cierre de las acciones  #
#################################################

def Lec(path):                                                                                   #Lectura de archivos .csv
    allFiles = glob.glob(path + "/*.csv")
    Data = []
    Completname = []
            
    for file_ in allFiles:
        Data.append(pd.read_csv(file_,index_col=None, header=0)) #skiprows=[int(0.3*len(file_))] selecciona el subconjunto de datos de entrenamiento
        #Completname= Completname.append(Completname,raw_input(file_))
    return(Data)
    

def ArcM(Data):                                                                                  #identifica el archivo con más filas
    filas=[]
    for i in range(len(Data)):
        filas.append(Data[i].shape[0])
    IndArchMax = filas.index(max(filas))
    return(IndArchMax)


def Sizequal(Data):                                                                             #Compara tamaños de archivos y los vuelve de igual tamaño
    for j in range(len(Data)):
        dif = Data[ArcM(Lec(path))].shape[0] - Data[j].shape[0]
        
        if dif != 0:
            Data[j] = Data[j].append([Data[ArcM(Lec(path))].iloc[:dif,:]]).reset_index(drop=True)
    return(Data)


def CompleT(Data):                                                                             #Completa cada dataFrame con la información adecuada
    for j in range(len(Data)):
        for i in range(Data[ArcM(Lec(path))].shape[0]):
            if Data[ArcM(Lec(path))].iloc[i, 0]!=Data[j].iloc[i, 0]:
                row=pd.DataFrame([[Data[ArcM(Lec(path))].iloc[i, 0], Data[j].iloc[i-1, 1],Data[j].iloc[i-1, 2],Data[j].iloc[i-1, 3], Data[j].iloc[i-1, 4],Data[j].iloc[i-1, 5],Data[j].iloc[i-1, 6]]], columns=['Fecha', 'Último', 'Apertura', 'Máximo', 'Mínimo', 'Vol.', '% var.'])
                Data[j] = pd.concat([Data[j].iloc[:i,:], row, Data[j].iloc[i:,:]]).reset_index(drop=True)
                Data[j]=Data[j].drop(Data[j].index[Data[ArcM(Lec(path))].shape[0]])
        
    return(Data)
 

def Sel(Data):                                                                                #Matriz de los precios de cierre de las acciones
    Pc=np.asarray([])
    flagFirst = 0
        
    for k in range(len(Data)):
        CloPrice  = Data[k].iloc[:,1].values    
        CloPrice = [s. replace('.','') for s in CloPrice] 
        CloPrice = [s. replace(',','.') for s in CloPrice]
        v = [float(s) for s in CloPrice] 
        
        if flagFirst==0:
            flagFirst = 1
            Pc = v
        else:
            Pc = np.vstack((Pc,v)) #"pega verticalmente"
    return(Pc)

Closing_Price=Sel(CompleT(Sizequal(Lec(path))))
#print(Closing_Price)
#print(np.shape(Closing_Price))

# In[ ]:


#Cálculo precios relativos

def retorn(M):
    M=np.transpose(M)
    vt=np.zeros((M.shape[0], M.shape[1]))
            
    for i in range(M.shape[1]):
        for j in range(M.shape[0]):
            vt[j][i] = M[j-1][i] /M[j][i] #Cálculo de vt (según artículo guía)
    
    Vt=np.transpose(np.delete(vt,0,0))
    onesColumn = np.ones((1,Vt.shape[1]));
    Vt = np.vstack((onesColumn,Vt)) # ubica elementos en la parte inferior - en este caso ubica filas-
    #plt.matshow(V_t)
    return Vt

# In[ ]:


#Cálculo promedio, matriz de covarianzas de los precios de cierre de cada acción

Me = np.mean(Closing_Price, axis = 1)
Va = np.var(Closing_Price[0,:], ddof = 1) # ddof es para que divida sobre (N - ddof)
Co = np.cov(Closing_Price)

# In[ ]:


#Cálculo promedio, matriz de covarianzas de los precios de cierre de cada acción

Me = np.mean(Closing_Price, axis = 1)
Va = np.var(Closing_Price[0,:], ddof = 1) # ddof es para que divida sobre (N - ddof)
Co = np.cov(Closing_Price)


# In[ ]:


#Solución analítica del cálculo de los pesos con el módelo clásico de Markowitz

from numpy import ndarray

def  wMa(rf,Pc):
    U= np.ones((Co.shape[0],1))
    E=np.transpose(np.array([Me]))
    A=np.matmul(np.transpose(U),np.matmul(np.linalg.inv(Co),E))
    B=np.matmul(np.transpose(E),np.matmul(np.linalg.inv(Co),E))
    C=np.matmul(np.transpose(U),np.matmul(np.linalg.inv(Co),U))
    D=B*C-A**2
    g=(1/D)*(B*np.matmul(np.linalg.inv(Co),U)-A*np.matmul(np.linalg.inv(Co),E))
    h=(1/D)*(C*np.matmul(np.linalg.inv(Co),E)-A*np.matmul(np.linalg.inv(Co),U))
    w=g+h*rf
    return w

print(wMa(0.01,Closing_Price))
print(np.sum(np.array([wMa(0.01,Closing_Price)])))

#plt.plot([1,2,3,4], [1,4,9,16], 'ro')
#plt.axis([0, 6, 0, 20])
#plt.title(i[:-4] + str(len(Vt)))
#plt.show()

# In[ ]:


#Cálculo del portafolio y la tasa de retorno

w_0=np.array([1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])  #Peso inicial según artículo
P_0=200                                                               #capital inicial a invertir

def Valorport(Vt,w,P_0):
   # P_(-1)=P_0
    w=w_0
    for i in range(Vt.shape[1]):
        P_i=P_(i-1)*np.matmul(Vt[:,i],w)     #Valor del portafolio en el periodo i
        if flagFirst==0:
            flagFirst = 1
            Pf= P_i
        else:
            Pf = np.vstack((Pf,P_i)) 
        print(Pf)    
        
        rho_i=P_i/P_(i-1) -1                #Tasa de retorno en t=i
        if flagFirst==0:
            flagFirst = 1
            rhof= rho_i
        else:
            rhof = np.vstack((rhof,rho_i))  
        print(rhof)     
        
        r_i=np.log(P_i/P_(i-1))                     #Tasa de retorno logarítmica
        if flagFirst==0:
            flagFirst = 1
            rf= r_i
        else:
            rf = np.vstack((rf,r_i))        
        print(rf)     
        w=wmv(Pc)
    return Pf,rhof,rf
