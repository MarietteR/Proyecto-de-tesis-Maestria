
# coding: utf-8

# In[ ]:


#Librerias
import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cvxopt import matrix, solvers

# In[ ]:


path ='/media/anmrodriguezsa/Datos/Dropbox/Universidad/Maestría en Actuaría y finanzas/Proyecto investigación Maestría/Python/acciones'

# In[ ]:


def Lec(path):                                                                           #Lectura de archivos .csv
    allFiles = glob.glob(path + "/*.csv")
    Data = []
    Completname = []
            
    for file_ in allFiles:                                  
        Completname.append(file_[144:-4])                                                #Vector de nombres de los archivos
        Data.append(pd.read_csv(file_, header=0, skiprows=range(1,int(0.3*len(file_))))) #skiprows=[int(0.3*len(file_))] selecciona el subconjunto de datos de entrenamiento
    return(Data,Completname)


def ArcM(Data):                                                                          #identifica el archivo con más filas
    filas=[]
    for i in range(len(Data)):
        filas.append(Data[i].shape[0])
    IndArchMax = filas.index(max(filas))
    return(IndArchMax)


def Sizequal(Data):                                                                     #Compara tamaños de archivos y los vuelve de igual tamaño
    for j in range(len(Data)):
        dif = Data[ArcM(Data)].shape[0] - Data[j].shape[0]
        
        if dif != 0:
            Data[j] = Data[j].append([Data[ArcM(Data)].iloc[:dif,:]]).reset_index(drop=True)
    return(Data)

def CompleT(Data):                                                                      #Completa cada dataFrame con la información adecuada
    for j in range(len(Data)):
        for i in range(Data[0].shape[0]):
            if Data[0].iloc[i, 0]!=Data[j].iloc[i, 0]:
                row=pd.DataFrame([[Data[0].iloc[i, 0], Data[j].iloc[i-1, 1],Data[j].iloc[i-1, 2],Data[j].iloc[i-1, 3], Data[j].iloc[i-1, 4],Data[j].iloc[i-1, 5],Data[j].iloc[i-1, 6]]], columns=['Fecha', 'Último', 'Apertura', 'Máximo', 'Mínimo', 'Vol.', '% var.'])
                Data[j] = pd.concat([Data[j].iloc[:i,:], row, Data[j].iloc[i:,:]]).reset_index(drop=True)
                Data[j]=Data[j].drop(Data[j].index[Data[0].shape[0]])
        
    return(Data)

 

def Sel(Data):                                                                          #Matriz de los precios de cierre de las acciones
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

Closing_Price=Sel(CompleT(Sizequal(Lec(path)[0])))
#print(Closing_Price)
#print(np.shape(Closing_Price))

# In[ ]:


#############################
# Cálculo precios relativos #
#############################

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


####################################################################################
# Media, Varianza, y Matriz de covarianzas de los precios de cierre de cada acción #
####################################################################################

Me = np.mean(Closing_Price, axis = 1)
Va = np.var(Closing_Price[0,:], ddof = 1) # ddof es para que divida sobre (N - ddof)
Co = np.cov(Closing_Price)


# In[ ]:

######################################################################################
# Función de optimización Modelo de Markowitz con restricción de pesos no negativos. #
######################################################################################

P = matrix(Co)                                      #Matriz de vaianzas y covarianzas
q = matrix(np.zeros((Co.shape[0],1)))               #matriz de ceros
A = matrix(np.asarray([Me,np.ones(Co.shape[0])]))   #Matriz de retornos de cada acción (1f) y unos (2f) 
b = matrix(np.array([[0.01],[1]]))                  #Matriz con las dos condiciones de igualdad
G_0 = matrix(-np.zeros((Co.shape[0],Co.shape[0])))  #Matriz ceros cuadrada 
G = matrix(-np.identity(Co.shape[0]))               #Matriz identidad negativa 
h = matrix(np.zeros((Co.shape[0],1)))               #Vector de ceros Dim(26,1)


sol=solvers.qp(P, q, G_0, h, A, b)

np.save('w.npy',sol['x'])
W_QPMkz=np.load('w.npy')
print(W_QPMkz)
print(np.sum(W_QPMkz))

# In[ ]:


##################################################################################
# Solución analítica del cálculo de los pesos con el módelo clásico de Markowitz #
##################################################################################

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
    # Prueba
    w1 = np.asarray([])
    for i in range(len(w)):
        w1 = np.append(w1, w[i][0])
    return w1

print(wMa(0.01,Closing_Price))
print(np.sum(np.array([wMa(0.01,Closing_Price)])))
print(Lec(path)[1])
print(type(wMa(0.01,Closing_Price)))
print(type(Lec(path)[1]))

plt.figure(figsize = ((25,20)))
label = np.arange(len(Lec(path)[1]))
plt.bar(label, wMa(0.01,Closing_Price))
plt.xticks(label, Lec(path)[1], fontsize=10, rotation=30)
plt.grid(b = True, which = 'both', linestyle = '--')

plt.show()

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
