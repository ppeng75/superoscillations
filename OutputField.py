from numba import njit
import numpy as np



def OutputField(lendom, waist, ET_thz, f)
    """
    Calculates output field over time domain for a given frequency
    :param lendom: Length of the domain (as a function)
    :param waist:  Waist size (as a function)
    :param ET_thz: Output field over space and time (as a function)
    :param f: Frequency of target output beam
    :param c0: User define parameter for shortening of coefficients
    :param g: Phenomenological loss (kappa/omega)
    """
    
    
###############################################################################
#
# Code to be executed
#
###############################################################################

c0 = ((n_thz[h]/n_ir)**2-1)/2/v_opt**2/tau**2 
g = kappa/omega			# 
    
    
L_period = lendom(freq[h],n_thz[h]) 	# period length (nm)
L_domain = L_period/2			# domain length (nm)
L_theo = 5e6 				# theoretical total length of the crystal (nm)
N_period = int(np.floor(L_theo/L_period)) # To get integer period number
N_domain = int(N_period*2)		# Need nearest integer domain
L_act = L_period*N_period 		# actual crystal length with integer number of periods (nm)
l = np.arange(0,L_act,L_domain/100)	# 1D array for trapz
Ethz = np.zeros((len(time),len(l)))	# Array stores at each time of the domain
L_w = 1/2*L_act  			# beam focus position inside PPLN (nm)
for i in range(0,len(time)):
    for k in range(0,len(l)):
        Ethz[i,k] = Et_thz(l[k],time[i],n_thz[h])
    
for i in range(0,len(Ethz_t)):
    for j in range(0,N_domain-1):
        Ethz_t[i,h] = Ethz_t[i,h]+(-1)**j*np.trapz(Ethz[i,(j)*100:(j+1)*100])

