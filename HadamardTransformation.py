import h5py
import scipy.io as sp
import libreria as p
import numpy as np
import matplotlib.pyplot as plt

titolo = "/workspaces/Hadamard/Data/480_SCRfluo_500_500msWH1_posneg_SPC_raw.mat"
base = p.Sequency(5)
f1 = h5py.File(titolo,'r')
print (f1.keys())

data = f1.get('spc')
data = np.array(data)
s= np.shape(data)
print(" dimension upn extraction: ", s)

#_____ preprocessing data_____
print("_____ preprocessing data _____")
## the first part of the measure is given by noise so we subtract it
print ("_____ Subtracting Noise _____")
data = data - data[0]
data = np.delete(data,0, axis=0)
## posneg
print("_____ Posneg _____")
data = p.posneg(data)
## it's a lot of data so we bin them
print ("_____ binning _____")
data = p.binning(data,16,2)
print ("dimension after binning  ",np.shape(data))

#riordino  
print ("----------riordino ---------")   
im = np.reshape(data,[32, 32, 1,-1],order='C')
print ("data shape  ",np.shape(im))
print ("base shape ",np.shape (base))
s = np.shape(im)
Tras  = np.zeros(s,dtype=float)

for i in range  (0,256):
    print ("  ",i, np.shape (im [:,:,0,i]), np.shape(base))
    R = np.dot(im [:,:,0,i],base)
    R = np.transpose(R)
    R = np.dot(R,base)
    Tras [:,:,0,i] = np.transpose(R)

x = np.sum (Tras,axis=-1)
plt.imsave('image_more.jpg', x[:,:,0])

x = p.binning(im,256,3)
x = x[:,:,0,0]
print ("x shape  ",np.shape(x))
R = np.dot(x,base)
R = np.transpose(R)
R = np.dot(R,base)
R = np.transpose(R)

plt.imsave('image_new.jpg', R)
