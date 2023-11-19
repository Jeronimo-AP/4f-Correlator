#Credit to youtube_channel/Python Metaphysics Series/vid29.ipynb on github and the book "Hands-on Image Processing with Python"
import numpy as np
from PIL import Image
import scipy as sp
import scienceplots
import scipy.fftpack as fp
from scipy.fft import fft2
from scipy.fft import ifft2
from scipy.fft import fftfreq
from scipy.fft import fftshift
import imageio
import cv2

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import animation
from matplotlib.animation import PillowWriter
import pint

plt.style.use(['science', 'notebook'])
u = pint.UnitRegistry()

#Define a function to get the field from the transform after propagating a certain distance
def compute_U_out(filtered, xv, yv, lam, z):
    kx = 2*np.pi * fftfreq(len(x), np.diff(x)[0])
    kxv, kyv = np.meshgrid(kx,kx)
    k = 2*np.pi/lam
    return fft2(filtered*np.exp(1j*z*np.sqrt(k**2-kxv**2-kyv**2))) #se calcula la fft2 de nuevo ya que queremos que se propage a través del segundo lente


#Define the colors for the plot
def wavelength_to_rgb(wavelength, gamma=0.8):
    wavelength = float(wavelength)
    if wavelength >= 380 and wavelength <= 440:
        attenuation = 0.3 + 0.7 * (wavelength - 380) / (440 - 380)
        R = ((-(wavelength - 440) / (440 - 380)) * attenuation) ** gamma
        G = 0.0
        B = (1.0 * attenuation) ** gamma
    elif wavelength >= 440 and wavelength <= 490:
        R = 0.0
        G = ((wavelength - 440) / (490 - 440)) ** gamma
        B = 1.0
    elif wavelength >= 490 and wavelength <= 510:
        R = 0.0
        G = 1.0
        B = (-(wavelength - 510) / (510 - 490)) ** gamma
    elif wavelength >= 510 and wavelength <= 580:
        R = ((wavelength - 510) / (580 - 510)) ** gamma
        G = 1.0
        B = 0.0
    elif wavelength >= 580 and wavelength <= 645:
        R = 1.0
        G = (-(wavelength - 645) / (645 - 580)) ** gamma
        B = 0.0
    elif wavelength >= 645 and wavelength <= 750:
        attenuation = 0.3 + 0.7 * (750 - wavelength) / (750 - 645)
        R = (1.0 * attenuation) ** gamma
        G = 0.0
        B = 0.0
    else:
        R = 0.0
        G = 0.0
        B = 0.0
    #R *= 255
    #G *= 255
    #B *= 255
    return (R, G, B)


#Define different wavelengths and lens focus
f = 1.3E8*u.nm
lam1 = 400*u.nm
lam2 = 505*u.nm
lam3 = 600*u.nm
lam4 = 700*u.nm

#Define custom colormaps to show the light
cmaps = [LinearSegmentedColormap.from_list('custom', 
                                            [(0,0,0),wavelength_to_rgb(lam.magnitude)],
                                            N=256) for lam in [lam1, lam2, lam3, lam4]]

#Calculation for a slit with green light
j="arrow"#input("Enter apature function" as a black and white image)#.upper()
U0 = np.array(Image.open(f'./images/{j}.jpg').convert('L'))
#U0 = cv2.flip(U0, 0)
#U0 = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
x = np.linspace(-23,23,U0.shape[0]) * u.mm
xv, yv = np.meshgrid(x, x)
freq = fft2(U0)
kx = fftfreq(len(x), np.diff(x)[0]) * 2 * np.pi # multiply by 2pi to get angular frequency
kxv, kyv = np.meshgrid(kx,kx)

#Filtering the Fourier plane for different filters
freq = fp.fft2(U0)
(w, h) = freq.shape
half_w, half_h = int(w/2), int(h/2)
freq1 = np.copy(freq)
freq2 = fp.fftshift(freq1)
freq_central = fp.fftshift(freq1)
freq_external = fp.fftshift(freq1)
freq_horizontal = fp.fftshift(freq1)
freq_vertical = fp.fftshift(freq1)
freq_edge_vertical = fp.fftshift(freq1)
freq_edge_horizontal = fp.fftshift(freq1)


# specify circle parameters: centre ij and radius
ci,cj=half_w,half_h
cr=15
# Create index arrays to z
c_x,c_y=np.meshgrid(np.arange(freq1.shape[0]),np.arange(freq1.shape[1]))

# calculate distance of all points to centre
dist=np.sqrt((c_x-ci)**2+(c_y-cj)**2)

# Assign value of 1 to those points where dist<cr:
freq_central[np.where(dist<cr)]=0 # select all but the first 20x20 (low) frequencies

freq_width=8

freq_horizontal[half_w-freq_width:half_w+freq_width+1,0:half_h*2] = 0 # select all but the horizontral frequencies
freq_vertical[0:half_w*2,half_h-freq_width:half_h+freq_width+1] = 0 # select all but the vertical frequencies
freq_external=freq_external - freq_central
freq_edge_vertical = freq_edge_vertical - freq_vertical #select only vertical frequencies
freq_edge_vertical[half_w-freq_width:half_w+freq_width+1,half_h-freq_width:half_h+freq_width+1] = 0 #select only vertical frecuencies but not central
freq_edge_horizontal = freq_edge_horizontal - freq_horizontal #select only horizontal frequencies
freq_edge_horizontal[half_w-freq_width:half_w+freq_width+1,half_h-freq_width:half_h+freq_width+1] = 0 #select only horizontal frecuencies but not central

im_i_c = np.clip(fp.fft2(fp.fftshift(freq_central)).real,0,None) # clip pixel values after IFFT
im_e_l_h = np.clip(fp.fft2(fp.fftshift(freq_horizontal)).real,0,None) 
im_e_l_v = np.clip(fp.fft2(fp.fftshift(freq_vertical)).real,0,None) 
im_e_c = np.clip(fp.fft2(fp.fftshift(freq_external)).real,0,None) 
im_edge_v = np.clip(fp.fft2(fp.fftshift(freq_edge_vertical)).real,0,None) 
im_edge_h = np.clip(fp.fft2(fp.fftshift(freq_edge_horizontal)).real,0,None) 

#Calculate the field after the second lens
U_new2 = compute_U_out(freq_central, xv, yv, lam=lam2, z=13*u.cm)
#U0 = cv2.flip(U0, 0)
#Plot all filters and images
fqmax=0.3
immax=1.5
fig, ax = plt.subplots(7, 2, figsize=(10,50))

ax[0,0].imshow(np.abs(U0), cmap=cmaps[1])
ax[0,1].imshow(np.abs(freq2)**0.45, cmap=cmaps[1])
ax[1,0].imshow(np.abs(freq_central)**fqmax, cmap=cmaps[1])
ax[1,1].imshow(im_i_c**immax, cmap=cmaps[1])
ax[2,0].imshow(np.abs(freq_horizontal)**fqmax, cmap=cmaps[1])
ax[2,1].imshow(im_e_l_h**immax, cmap=cmaps[1])
ax[3,0].imshow(np.abs(freq_vertical)**fqmax, cmap=cmaps[1])
ax[3,1].imshow(im_e_l_v**immax, cmap=cmaps[1])
ax[4,0].imshow(np.abs(freq_external)**fqmax, cmap=cmaps[1])
ax[4,1].imshow(im_e_c**immax, cmap=cmaps[1])
ax[5,0].imshow(np.abs(freq_edge_vertical)**fqmax, cmap=cmaps[1])
ax[5,1].imshow(im_edge_v**immax, cmap=cmaps[1])
ax[6,0].imshow(np.abs(freq_edge_horizontal)**fqmax, cmap=cmaps[1])
ax[6,1].imshow(im_edge_h**immax, cmap=cmaps[1])

ax[0,0].set_title("Apertura")
ax[0,1].set_title("Transformada de Fourier")
ax[1,0].set_title("Filtro circular (interno)")
ax[1,1].set_title("Imagen")
ax[2,0].set_title("Filtro horizontal")
ax[2,1].set_title("Imagen")
ax[3,0].set_title("Filtro vertical")
ax[3,1].set_title("Imagen")
ax[4,0].set_title("Filtro circular (externo)")
ax[4,1].set_title("Imagen")
ax[5,0].set_title("Filtro borde horizontal")
ax[5,1].set_title("Imagen")
ax[6,0].set_title("Filtro borde vertical")
ax[6,1].set_title("Imagen")

ax[0,0].axis('off')
ax[0,1].axis('off')
ax[1,0].axis('off')
ax[1,1].axis('off')
ax[2,0].axis('off')
ax[2,1].axis('off')
ax[3,0].axis('off')
ax[3,1].axis('off')
ax[4,0].axis('off')
ax[4,1].axis('off')
ax[5,0].axis('off')
ax[5,1].axis('off')
ax[6,0].axis('off')
ax[6,1].axis('off')

plt.savefig(f'./plots/{j}_sim.png')  
