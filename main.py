import numpy as np
from wdata_io import WData, Var
import math
from numba import njit #  increases performance

@njit
def f(k, lmax): # f(k) potential in Fourier space
    if k == 0:
        return 6 * np.pi * lmax**2
    else:
        return 4 * np.pi * (1 - np.cos(k * lmax * np.sqrt(3)))/k**2

@njit
def calculate_fk(ks,f_k):
    for ix, kx in enumerate(ks):
        for iy, ky in enumerate(ks):
            for iz, kz in enumerate(ks):
                f_k[ix, iy, iz] = f(np.sqrt(kx**2 + ky**2 + kz**2), L_max)*e2
    return f_k


def distribute_densities(it):

    #* Firstly I'm distributing densities in 
    #* a normal size lattice LxLyLz:

    rho3 = np.zeros((Nx, Ny, Nz))
    for ix in range(Nx):
        for iy in range(Ny):
            for iz in range(Nz):
                rho3[ix, iy, iz] = data.density_p[it, ix, iy, iz]

    #* Here I extend the lattice in every dimension: 

    pad = ((math.floor((N3_max - Nx)/2), math.ceil((N3_max - Nx)/2)), (math.floor((N3_max - Ny)/2),
           math.ceil((N3_max - Ny)/2)), (math.floor((N3_max - Nz)/2), math.ceil((N3_max - Nz)/2)))

    return np.pad(rho3, pad, 'constant', constant_values=0) # 27LxLyLz lattice is returned


def calculate_coulomb_energy():
    return np.sum(np.multiply(coul3.real, rho3)) * dxyz / 2


file_wtxt = None #! NAME OF FILE *.wtxt
fwtxt = None #! PATH TO *wtxt file path + file_wtxt

data = WData.load(fwtxt, check_data=False) 
dxyz = data.dxyz[0]*data.dxyz[1]*data.dxyz[2]

e2 = 197.3269631 / 137.035999679

dx, dy, dz = data.dxyz[0], data.dxyz[1], data.dxyz[2] # lattice constants
Nx, Ny, Nz = data.Nxyz[0], data.Nxyz[1], data.Nxyz[2] # numbers of equidistant lattice points

N_max = np.max(data.Nxyz)
N3_max = 3 * N_max
d_max = np.max(data.dxyz)
L_max = N_max*d_max
ks = 2*np.pi*(np.fft.fftfreq(N3_max, d=d_max)) # DFT sample frequencies

fk = np.zeros((N3_max, N3_max, N3_max), dtype=np.complex128)

print(f'{" CALCULATING f(k) ":.<30}')
f_k = calculate_fk(ks,fk)
print(f'{" DONE CALCULATING f(k) ":.<30}')

for it in range(0,data.N,10):  # data.Nt):
    
    rho3 = distribute_densities(it)
    rho3_k = np.fft.fftn(rho3,s = (N3_max, N3_max, N3_max), axes = (0,1,2)) #FFT OVER 27LxLyLz lattice
    coul3_k = np.zeros((N3_max, N3_max, N3_max), dtype=np.complex128)
    np.multiply(rho3_k, f_k, out=coul3_k)

    coul3 = np.float64(np.fft.ifftn(coul3_k, s = (N3_max,N3_max,N3_max),axes = (0,1,2))) # Inverse FFT
    
    ecoul = calculate_coulomb_energy()

    with open(f"output.txt", 'a+') as file_out:
            print(f'{it}, {ecoul}', file=file_out)