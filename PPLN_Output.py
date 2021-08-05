from numba import njit
import matplotlib.pyplot as plt
import numpy as np
import _pickle as pickle
from multiprocessing import Pool
from OutputField import OutputField
import os



####################################################################################################
#
# Declare functions to pass to class
#
####################################################################################################


@njit
def lendom(f,nthz):
    """
    Calculates the theoretical domain length
    """
    return c/f*1/(abs(n_ir-nthz))


@njit
def waist(z):
    """
    Calculates the Waist size at distance z from focus point
    """
    return w0*np.sqrt(1+(abs(z-L_w)/z_R)**2)    


@njit
def Et_thz(z,t,nthz):
    """
    Calculates output field at position z at time t
    """

    coef = (1+w0*c0)/(1+waist(z)*c0)
    A = tau**2/4+g/c*(L_act-z)
    B = (n_ir*z+nthz*(L_act-z))/c
    Et = A0*tau*np.sqrt(PI)/(A**(3/2))*np.exp(-(B-t)**2/(4*A))*(1-(B-t)**2/(2*A))
    return Et*coef

####################################################################################################
#
# Define physical constants
#
####################################################################################################

F = 150. # focus of convex lense (mm)
d_beam = 10. # beam diameter (mm)
wl = 1030e-6 # wavelength of incident laser (mm)
c = 3e5 # (mm/ps)
n_ir = 2.23  #index for laser beam inside PPLN (no units)
n_thz = np.array([5.11,5.11,5.11,5.12,5.13,5.13,5.14,5.15,5.17,5.31]) #index for THz beam inside PPLN (no units)
tau = 190e-3  #laser pulse duration (ps)
A0 = 1. #laser amplitude (mm)
g = 7.5e-3*2 # g = kappa/omega, this particular value is used in most of the literatures (ps)
PI = np.pi
time = np.arange(0,100.01,0.05) # (ps)
freq = np.array([0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,2.0]) 			# chosen frequencies (THz)
alpha = np.array([1000,1150,1300,1450,1600,1750,1900,2050,2500,8000])*1e-3 	# attenuation (mm^-1)
kappa = (alpha*c/freq)/4/np.pi 						# (unitless)
omega = freq*2*PI								# (2pi THz)
w0 = 4/PI*wl*F/d_beam  							# beam waist diameter at focus point (nm)
z_R = PI*w0**2*n_ir/wl 							# Rayleigh distance (mm)
v_opt = c/n_ir									# Fundamental ir

####################################################################################################
#
# Declare path for saving and archive of data
#
####################################################################################################

def Replace(str1):
    str1 = str1.replace('.', ',')
    return str1

if not os.path.exists('./Data_Archive'):
    os.mkdir('./Data_Archive')

Saves_path = './Data_Archive'
filename = 'PPLN_Output-Focus' + Replace(str(F)) + '_BeamDiameter' + Replace(str(d_beam))
path = os.path.join(Saves_path, filename)
os.mkdir(path)

# Create a text file to store quick reference information. Uncomment if needed.
#outfile = filename + '.txt'
#outpath = os.path.join(path, outfile)
#out = open(path, 'w')
#out.write(
#
#)         

####################################################################################################
#
# calculate theoretical field, time domain results are saved in variable Ethz_t
#
####################################################################################################



Ethz_t = np.zeros((len(time),len(freq))) 	# This is what the class will return, 2D array, Field amplitude:(axis 0: Time, axis 1: Frequency)

# List of things we need
# v_thz[h], gamma[h], c0[h], and g[h] will need to be callable, so we will need something to represent h
# For calculation of L_period, need to implement callable h from previous step

for h in range(0,len(freq)):
    v_thz = c/n_thz[h] 			# this calc can be done elementwise
    gamma = n_thz[h]/n_ir 			# This calc can be done elementwise
    c0 = (gamma**2-1)/2/v_opt**2/tau**2 	# may not be necessary
    # g = kappa[h]/omega[h]			# use numpy for automatic 
    g = kappa/omega
    
    
    L_period = lendom(freq[h],n_thz[h]) 	# period length (nm)
    L_domain = L_period/2			# domain length (nm)
    L_theo = 5. 				# theoretical total length of the crystal (mm)
    N_period = int(np.floor(L_theo/L_period)) # To get integer period number
    N_domain = int(N_period/2)		# Need nearest integer domain (unsure of *2 or /2)
    L_act = L_period*N_period 		# actual crystal length with integer number of periods (mm)
    l = np.arange(0,L_act,L_domain/100)	# Position array for holding z.
    Ethz = np.zeros((len(time),len(l)))	# Array stores at each time of the domain
    L_w = 1/2*L_act  				# beam focus position inside PPLN (mm)
    for i in range(0,len(time)):
        for k in range(0,len(l)):
            Ethz[i,k] = Et_thz(l[k],time[i],n_thz[h])
    
    for i in range(0,len(Ethz_t)):
        for j in range(0,N_domain-1):
            Ethz_t[i,h] = Ethz_t[i,h]+(-1)**j*np.trapz(Ethz[i,(j)*100:(j+1)*100])
            
    '''
    to plot results, not necessary
    
    '''
        
    Ethz_t[:,h] = Ethz_t[:,h]/L_act
    [freq0,Ethz_f] = arfft(time,Ethz_t[:,h],2)
    plt.figure()
    # mpl.pyplot.plot(time,Ethz)
    plt.plot(time*1e12,Ethz_t[:,h],label='Period='+str(int(L_period*1e6))+' um')
    # plt.xlim([0,10])
    # plt.ylim([-3e16,3e16])
    plt.xlabel('Time (ps)')
    plt.ylabel('Amplitude (a.u.)')
    plt.legend(loc='best')
    plt.figure()
    plt.plot(freq0,abs(Ethz_f),label='Period='+str(int(L_period*1e6))+' um')
    plt.xlabel('frequency (Hz)')
    plt.xlim([0,5e12])
    plt.ylabel('Amplitude (a.u.)')
    plt.legend(loc='best')
