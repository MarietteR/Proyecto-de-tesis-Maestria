
# coding: utf-8

# ## Reconstrucción

# In[3]:


#Librerias
import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import quandl as qd
from scipy.optimize import minimize
import statistics as st


# In[4]:


path ='/media/anmrodriguezsa/Datos/Dropbox/Universidad/Maestría en Actuaría y finanzas/Proyecto investigación Maestría/Python/acciones'  ##Linux
#path ='C:/Users/Ángela/Desktop/Proyecto1/acciones'   #windows


# In[7]:


def LecYSel(path):
    allFiles = glob.glob(path + "/*.csv")
    flagFirst = 0
    
    def Data(k):
        Data=np.asarray([])
        Data= pd.read_csv(k,index_col=None, header=0)  #Lectura de archivos .csv
        return(Data)
    
    Data(allFiles)
        
    for k in allFiles:
        Data_(k)=np.asarray([])
        Data(k)
        print(Data(k))
        Data_(k)=Data(k).convert_objects(convert_numeric=True).dtypes
        while Data_(k).shape[0]!=Data_(k+1).shape[0]:
            for i in range(Data(k).shape[0]):
                if Data(k)[i][0]!=Data(k+1)[i][0]:
                    Data(k+1)[i][0]=np.insert(Data(k+1), i, np.array((Data(k)[i][0], Data(k+1)[i][1],Data(k+1)[i][2], Data(k+1)[i][3], Data(k+1)[i][4], Data(k+1)[i][5],Data(k+1)[i][5])), 0)     
LecYSel(path)


# In[14]:


# Lectura de archivos .csv y Selección de los precios de cierre de cada acción.

def LecYSel(path):
    allFiles = glob.glob(path + "/*.csv")
    Pc=np.asarray([])
    flagFirst = 0
        
    for k in allFiles:
        Data = pd.read_csv(k,index_col=None, header=0)  #Lectura de archivos .csv           
        x  = Data["Último"].values   #Selección de los valores del precio de cierre de las acciones
        x = [s. replace('.','') for s in x] 
        x = [s. replace(',','.') for s in x]
        v = [float(s) for s in x] 
        
        if flagFirst==0:
            flagFirst = 1
            Pc = v
        else:
            Pc = np.vstack((Pc,v)) #"pega"
    return(Pc)

LecYSel(path)


# In[30]:


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


# In[24]:


#Cálculo promedio, desviación estándar, matriz de covarianzas de los precios de cierre de cada acción

def Av(X):                      
    Av=np.zeros(X.shape[0])           
    
    for i in range(X.shape[0]):
        Av[i]= st.mean(X[i,:])        #Cálculo de la media 
    return Av

def cov(a, b):
    if len(a) != len(b):
        return

    a_mean = np.mean(a)
    b_mean = np.mean(b)
    sum = 0

    for i in range(0, len(a)):
        sum += ((a[i] - a_mean) * (b[i] - b_mean))     #Cálculo covarianza entre dos vectores
    return sum/(len(a)-1)


def Mcov(M):
    C=np.zeros((M.shape[0],M.shape[0]))
 
    for i in range(M.shape[0]):
        for j in range(M.shape[0]):
            C[i][j] = cov(M[i,:],M[j,:])                            #Matriz de covarianza
    return(C)


# In[26]:


#Simulación del cálculo de los pesos con el módelo clásico de Markowitz

def  wM(rf,Pc):
    M= Mcov(Pc)
    U= np.ones(Pc.shape[0])
    A=np.matmul(np.transpose(U),np.linalg.inv(M),Av(Pc))
    B=np.matmul(np.transpose(U),np.linalg.inv(M),U)
    C=np.matmul(np.transpose(Av(Pc)),np.linalg.inv(M),Av(Pc))
    D=B*C-A**2
    g=(1/D)*(B*np.matmul(np.linalg.inv(M),U)-A*np.matmul(np.linalg.inv(M),Av(Pc)))
    h=(1/D)*(C*np.matmul(np.linalg.inv(M),Av(Pc))-A*np.matmul(np.linalg.inv(M),U))
    w=g+h*rf
    
    wN=np.zeros(len(w))
    wS=np.sum(abs(w))
    for i in range(len(w)):        
        wN[i]=abs(w[i])/wS
    
    return w,wN

#wM(0,LecYSel(path))

#Simulación del cálculo de los pesos con el módelo mínima varianza

def  wmv(Pc):
    M= Mcov(Pc)
    U= np.ones(Pc.shape[0])
    C=np.matmul(np.transpose(Av(Pc)),np.linalg.inv(M),Av(Pc))
    w=(1/C)*np.matmul(np.linalg.inv(M),U)
    
    #wN=np.zeros(len(w))
    #wS=np.sum(abs(w))
    #for i in range(len(w)):        
     #   wN[i]=abs(w[i])/wS
    
    return w


# In[32]:


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

