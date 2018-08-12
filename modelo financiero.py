
# coding: utf-8

# In[ ]:


#Librerias
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statistics as st

# In[ ]:


path ='/media/anmrodriguezsa/DA4E639F4E6372E5/Users/Ángela/Desktop/Proyecto1/acciones'  ##Linux
#path ='C:/Users/Ángela/Desktop/Proyecto1/acciones'   #windows

# In[ ]:


# Lectura de archivos .csv y Selección de los precios de cierre de cada acción.

def LecYSel(path):
    allFiles = glob.glob(path + "/*.csv")
    Pc=np.asarray([])
    flagFirst = 0
        
    for file_ in allFiles:
        Data = pd.read_csv(file_,index_col=None, header=0)  #Lectura de archivos .csv
        x  = Data["Último"].values   #Selección de los valores del precio de cierre de las acciones
        x = [s. replace('.','') for s in x] 
        x = [s. replace(',','.') for s in x]
        v = [float(s) for s in x] 
        closingPrice = np.flipud(v)
        
        if flagFirst==0:
            flagFirst = 1
            Pc = closingPrice
        else:
            Pc = np.vstack((Pc,closingPrice)) #"transpone"
    return(Pc)

# In[ ]:


#Cálculo promedio, desviación estándar, matriz de covarianzas de los precios de cierre de cada acción

def Av(X):                      
    Av=np.zeros(X.shape[0])           
    
    for i in range(X.shape[0]):
        Av[i]= st.mean(X[i,:])        #Cálculo de la media 
    return Av

def Dst(X):                      
    DS=np.zeros(X.shape[0])           
    
    for i in range(X.shape[0]):
        DS[i]= st.stdev(X[i,:])       #Cálculo de Desviación Estándar
    return DS

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


y_t=Vt[:,0] #Vector de precio relativo

w_0=np.array([1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])

p_0=200    #capital inicial a invertir
p_1=p_0*np.matmul(y_t,w_0) #Valor del portafolio en el periodo 1
rho_1=p_1/p_0 -1     #Tasa de retorno
print(rho_1)
r_1=np.log(p_1/p_0)   #Tasa de retorno logarítmica
print(r_1)
