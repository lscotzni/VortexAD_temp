import numpy as np 
import csdl_alpha as csdl

from VortexAD.utils.atan2_switch import atan2_switch

def compute_doublet_influence(dpij, mij, ek, hk, rk, dx, dy, dz, mu=1., mode='potential'):
    # each input is a list of length 4, holding the values for the corners
    # ex: mij = [mij_1. mij_2, mij_3, mij_4]
    if mode == 'potential':
        doublet_influence = mu/4/np.pi*(
            csdl.arctan((mij[0]*ek[0]-hk[0])/(dz[0]*rk[0]+1.e-12)) - csdl.arctan((mij[0]*ek[1]-hk[1])/(dz[0]*rk[1]+1.e-12)) + 
            csdl.arctan((mij[1]*ek[1]-hk[1])/(dz[1]*rk[1]+1.e-12)) - csdl.arctan((mij[1]*ek[2]-hk[2])/(dz[1]*rk[2]+1.e-12)) + 
            csdl.arctan((mij[2]*ek[2]-hk[2])/(dz[2]*rk[2]+1.e-12)) - csdl.arctan((mij[2]*ek[3]-hk[3])/(dz[2]*rk[3]+1.e-12)) + 
            csdl.arctan((mij[3]*ek[3]-hk[3])/(dz[3]*rk[3]+1.e-12)) - csdl.arctan((mij[3]*ek[0]-hk[0])/(dz[3]*rk[0]+1.e-12))
            # csdl.arctan((mij[:,:,:,0]*ek[:,:,:,0]-hk[:,:,:,0])/(dz[:,:,:,0]*rk[:,:,:,0]+1.e-12)) - csdl.arctan((mij[:,:,:,0]*ek[:,:,:,1]-hk[:,:,:,1])/(dz[:,:,:,0]*rk[:,:,:,1]+1.e-12)) + 
            # csdl.arctan((mij[:,:,:,1]*ek[:,:,:,1]-hk[:,:,:,1])/(dz[:,:,:,1]*rk[:,:,:,1]+1.e-12)) - csdl.arctan((mij[:,:,:,1]*ek[:,:,:,2]-hk[:,:,:,2])/(dz[:,:,:,1]*rk[:,:,:,2]+1.e-12)) + 
            # csdl.arctan((mij[:,:,:,2]*ek[:,:,:,2]-hk[:,:,:,2])/(dz[:,:,:,2]*rk[:,:,:,2]+1.e-12)) - csdl.arctan((mij[:,:,:,2]*ek[:,:,:,3]-hk[:,:,:,3])/(dz[:,:,:,2]*rk[:,:,:,3]+1.e-12)) + 
            # csdl.arctan((mij[:,:,:,3]*ek[:,:,:,3]-hk[:,:,:,3])/(dz[:,:,:,3]*rk[:,:,:,3]+1.e-12)) - csdl.arctan((mij[:,:,:,3]*ek[:,:,:,0]-hk[:,:,:,0])/(dz[:,:,:,3]*rk[:,:,:,0]+1.e-12))
            # np.arctan2((mij[:,0]*ek[:,0]-hk[:,0]),(dk[:,0,2]*rk[:,0]+1.e-12)) - np.arctan2((mij[:,0]*ek[:,1]-hk[:,1]),(dk[:,0,2]*rk[:,1]+1.e-12)) + 
            # np.arctan2((mij[:,1]*ek[:,1]-hk[:,1]),(dk[:,1,2]*rk[:,1]+1.e-12)) - np.arctan2((mij[:,1]*ek[:,2]-hk[:,2]),(dk[:,1,2]*rk[:,2]+1.e-12)) + 
            # np.arctan2((mij[:,2]*ek[:,2]-hk[:,2]),(dk[:,2,2]*rk[:,2]+1.e-12)) - np.arctan2((mij[:,2]*ek[:,3]-hk[:,3]),(dk[:,2,2]*rk[:,3]+1.e-12)) + 
            # np.arctan2((mij[:,3]*ek[:,3]-hk[:,3]),(dk[:,3,2]*rk[:,3]+1.e-12)) - np.arctan2((mij[:,3]*ek[:,0]-hk[:,0]),(dk[:,3,2]*rk[:,0]+1.e-12))
        ) # note that dk[:,i,2] is the same for all i
        return doublet_influence
    elif mode == 'velocity':
        return