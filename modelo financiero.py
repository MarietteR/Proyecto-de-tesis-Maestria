
# coding: utf-8

# In[ ]:


#Librerias
import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import quandl as qd
from scipy.optimize import minimize
import statistics as st

# In[ ]:


path ='/media/anmrodriguezsa/Datos/Dropbox/Universidad/Maestría en Actuaría y finanzas/Proyecto investigación Maestría/Python/acciones'  ##Linux
#path ='C:/Users/Ángela/Desktop/Proyecto1/acciones'   #windows

# In[ ]:


#DEfinición matriz de precios de cierre de las acciones

def Lec(path):           #Lectura de archivos .csv
    allFiles = glob.glob(path + "/*.csv")
    Data = []
            
    for file_ in allFiles:
        Data.append(pd.read_csv(file_,index_col=None, header=0))
    return(Data)

#Lec(path)
#print(Lec(path))

def ArcM(Data):   #identifica el archivo con más filas
    filas=[]
    for i in range(len(Data)):
        filas.append(Data[i].shape[0])
    IndArchMax = filas.index(max(filas))
    return(IndArchMax)


def Sizequal(Data):        #Compara tamaños de archivos y los vuelve de igual tamaño
    for j in range(len(Data)):
        dif = Data[ArcM(Lec(path))].shape[0] - Data[j].shape[0]
        
        if dif != 0:
            Data[j] = Data[j].append([Data[ArcM(Lec(path))].iloc[:dif,:]]).reset_index(drop=True)
    return(Data)

#Sizequal(Lec(path))
#print(Sizequal(Lec(path)))

def CompleT(Data):        #Completa cada dataFrame con la información adecuada
    for j in range(len(Data)):
        for i in range(Data[ArcM(Lec(path))].shape[0]):
            if Data[ArcM(Lec(path))].iloc[i, 0]!=Data[j].iloc[i, 0]:
                row=pd.DataFrame([[Data[ArcM(Lec(path))].iloc[i, 0], Data[j].iloc[i-1, 1],Data[j].iloc[i-1, 2],Data[j].iloc[i-1, 3], Data[j].iloc[i-1, 4],Data[j].iloc[i-1, 5],Data[j].iloc[i-1, 6]]], columns=['Fecha', 'Último', 'Apertura', 'Máximo', 'Mínimo', 'Vol.', '% var.'])
                Data[j] = pd.concat([Data[j].iloc[:i,:], row, Data[j].iloc[i:,:]]).reset_index(drop=True)
                Data[j]=Data[j].drop(Data[j].index[Data[ArcM(Lec(path))].shape[0]])
        
    return(Data)

#CompleT(Sizequal(Lec(path)))
#print(CompleT(Sizequal(Lec(path)))) 

def Sel(Data):         #Selección de los valores del precio de cierre de las acciones
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

Sel(CompleT(Sizequal(Lec(path))))
print(Sel(CompleT(Sizequal(Lec(path)))))

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


# In[ ]:


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
    
    wN=np.zeros(len(w))
    wS=np.sum(abs(w))
    for i in range(len(w)):        
        wN[i]=abs(w[i])/wS
    
    return w,wN

wmv(LecYSel(path))

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
