import numpy as np 
import csdl_alpha as csdl

from VortexAD.utils.atan2_switch import atan2_switch

def compute_doublet_influence(dpij, mij, ek, hk, rk, dx, dy, dz, mu=1., mode='potential'):
    # each input is a list of length 4, holding the values for the corners
    # ex: mij = [mij_1. mij_2, mij_3, mij_4]
    if mode == 'potential':
        t1_y = dz[0]*dpij[0][0] * ((ek[0]*dpij[0][1]-hk[0]*dpij[0][0])*rk[1] - (ek[1]*dpij[0][1]-hk[1]*dpij[0][0])*rk[0])
        t2_y = dz[1]*dpij[1][0] * ((ek[1]*dpij[1][1]-hk[1]*dpij[1][0])*rk[2] - (ek[2]*dpij[1][1]-hk[2]*dpij[1][0])*rk[1])
        t3_y = dz[2]*dpij[2][0] * ((ek[2]*dpij[2][1]-hk[2]*dpij[2][0])*rk[3] - (ek[3]*dpij[2][1]-hk[3]*dpij[2][0])*rk[2])
        t4_y = dz[3]*dpij[3][0] * ((ek[3]*dpij[3][1]-hk[3]*dpij[3][0])*rk[0] - (ek[0]*dpij[3][1]-hk[0]*dpij[3][0])*rk[3])

        t1_x = dz[0]**2*rk[0]*rk[1]*dpij[0][0]**2 + (ek[0]*dpij[0][1]-hk[0]*dpij[0][0])*(ek[1]*dpij[0][1]-hk[1]*dpij[0][0])
        t2_x = dz[1]**2*rk[1]*rk[2]*dpij[1][0]**2 + (ek[1]*dpij[1][1]-hk[1]*dpij[1][0])*(ek[2]*dpij[1][1]-hk[2]*dpij[1][0])
        t3_x = dz[2]**2*rk[2]*rk[3]*dpij[2][0]**2 + (ek[2]*dpij[2][1]-hk[2]*dpij[2][0])*(ek[3]*dpij[2][1]-hk[3]*dpij[2][0])
        t4_x = dz[3]**2*rk[3]*rk[0]*dpij[3][0]**2 + (ek[3]*dpij[3][1]-hk[3]*dpij[3][0])*(ek[0]*dpij[3][1]-hk[0]*dpij[3][0])

        atan2_1 = 2*csdl.arctan(t1_y / ((t1_x**2 + t1_y**2 + 1.e-12)**0.5 + t1_x)) # 1
        # atan2_2 = 2*csdl.arctan(t2_y / ((t2_x**2 + t2_y**2 + 1.e-12)**0.5 + t2_x)) # 2 
        atan2_3 = 2*csdl.arctan(t3_y / ((t3_x**2 + t3_y**2 + 1.e-12)**0.5 + t3_x)) # 3
        # atan2_4 = 2*csdl.arctan(t4_y / ((t4_x**2 + t4_y**2 + 1.e-12)**0.5 + t4_x)) # 4

        # atan2_1 = 2*csdl.arctan(((t1_x**2 + t1_y**2)**0.5 - t1_x) / (t1_y + 1.e-12)) # 5
        atan2_2 = 2*csdl.arctan(((t2_x**2 + t2_y**2)**0.5 - t2_x) / (t2_y + 1.e-12)) # 6
        # atan2_3 = 2*csdl.arctan(((t3_x**2 + t3_y**2)**0.5 - t3_x) / (t3_y + 1.e-12)) # 7
        atan2_4 = 2*csdl.arctan(((t4_x**2 + t4_y**2)**0.5 - t4_x) / (t4_y + 1.e-12)) # 8

        # NOTE-S
        # 1-3-6-8 works compared to Jiayao's code
        # test other combinations

        doublet_potential = mu/4/np.pi*(
            csdl.arctan((mij[0]*ek[0]-hk[0])/(dz[0]*rk[0]+1.e-12)) - csdl.arctan((mij[0]*ek[1]-hk[1])/(dz[0]*rk[1]+1.e-12)) + 
            csdl.arctan((mij[1]*ek[1]-hk[1])/(dz[1]*rk[1]+1.e-12)) - csdl.arctan((mij[1]*ek[2]-hk[2])/(dz[1]*rk[2]+1.e-12)) + 
            csdl.arctan((mij[2]*ek[2]-hk[2])/(dz[2]*rk[2]+1.e-12)) - csdl.arctan((mij[2]*ek[3]-hk[3])/(dz[2]*rk[3]+1.e-12)) + 
            csdl.arctan((mij[3]*ek[3]-hk[3])/(dz[3]*rk[3]+1.e-12)) - csdl.arctan((mij[3]*ek[0]-hk[0])/(dz[3]*rk[0]+1.e-12))
        ) # note that dk[:,i,2] is the same for all i

        doublet_potential = mu/4/np.pi*( atan2_1 + atan2_2 + atan2_3 + atan2_4) # note that dk[:,i,2] is the same for all i

        return doublet_potential
    elif mode == 'velocity':
        core_size = 1.e-4
        doublet_velocity = csdl.Variable(shape=dx[0].shape + (3,), value=0.)
        r1r2 = rk[0]*rk[1]
        r2r3 = rk[1]*rk[2]
        r3r4 = rk[2]*rk[3]
        r4r1 = rk[3]*rk[0]
        
        u = mu/(4*np.pi) * (
            (dz[0] * -dpij[0][1] * (rk[0]+rk[1]))/(r1r2*(r1r2-(dx[0]*dx[1]+dy[0]*dy[1]+dz[0]*dz[1])) + core_size) + 
            (dz[1] * -dpij[1][1] * (rk[1]+rk[2]))/(r2r3*(r2r3-(dx[1]*dx[2]+dy[1]*dy[2]+dz[1]*dz[2])) + core_size) + 
            (dz[2] * -dpij[2][1] * (rk[2]+rk[3]))/(r3r4*(r3r4-(dx[2]*dx[3]+dy[2]*dy[3]+dz[2]*dz[3])) + core_size) + 
            (dz[3] * -dpij[3][1] * (rk[3]+rk[0]))/(r4r1*(r4r1-(dx[3]*dx[0]+dy[3]*dy[0]+dz[3]*dz[0])) + core_size) 
        )

        v = mu/(4*np.pi) * (
            (dz[0] * dpij[0][0] * (rk[0]+rk[1]))/(r1r2*(r1r2-(dx[0]*dx[1]+dy[0]*dy[1]+dz[0]*dz[1])) + core_size) + 
            (dz[1] * dpij[1][0] * (rk[1]+rk[2]))/(r2r3*(r2r3-(dx[1]*dx[2]+dy[1]*dy[2]+dz[1]*dz[2])) + core_size) + 
            (dz[2] * dpij[2][0] * (rk[2]+rk[3]))/(r3r4*(r3r4-(dx[2]*dx[3]+dy[2]*dy[3]+dz[2]*dz[3])) + core_size) + 
            (dz[3] * dpij[3][0] * (rk[3]+rk[0]))/(r4r1*(r4r1-(dx[3]*dx[0]+dy[3]*dy[0]+dz[3]*dz[0])) + core_size) 
        )

        w = mu/(4*np.pi) * (
            (2*dz[0]**2*dpij[0][1]*dpij[0][0]*rk[0] - (dpij[0][1]*ek[0]-hk[0]*dpij[0][0])*(rk[0]+dz[0]**2/(rk[0] + core_size))*dpij[0][0]) / ((dz[0]*rk[0]*dpij[0][0])**2 + (dpij[0][1]*ek[0]-hk[0]*dpij[0][0])**2 + core_size) - # Term 1
            (2*dz[1]**2*dpij[0][1]*dpij[0][0]*rk[1] - (dpij[0][1]*ek[1]-hk[1]*dpij[0][0])*(rk[1]+dz[1]**2/(rk[1] + core_size))*dpij[0][0]) / ((dz[0]*rk[1]*dpij[0][0])**2 + (dpij[0][1]*ek[1]-hk[1]*dpij[0][0])**2 + core_size) + # Term 2

            (2*dz[1]**2*dpij[1][1]*dpij[1][0]*rk[1] - (dpij[1][1]*ek[1]-hk[1]*dpij[1][0])*(rk[1]+dz[1]**2/(rk[1] + core_size))*dpij[1][0]) / ((dz[1]*rk[1]*dpij[1][0])**2 + (dpij[1][1]*ek[1]-hk[1]*dpij[1][0])**2 + core_size) - # Term 3
            (2*dz[2]**2*dpij[1][1]*dpij[1][0]*rk[2] - (dpij[1][1]*ek[2]-hk[2]*dpij[1][0])*(rk[2]+dz[2]**2/(rk[2] + core_size))*dpij[1][0]) / ((dz[1]*rk[2]*dpij[1][0])**2 + (dpij[1][1]*ek[2]-hk[2]*dpij[1][0])**2 + core_size) + # Term 4

            (2*dz[2]**2*dpij[2][1]*dpij[2][0]*rk[2] - (dpij[2][1]*ek[2]-hk[2]*dpij[2][0])*(rk[2]+dz[2]**2/(rk[2] + core_size))*dpij[2][0]) / ((dz[2]*rk[2]*dpij[2][0])**2 + (dpij[2][1]*ek[2]-hk[2]*dpij[2][0])**2 + core_size) - # Term 5
            (2*dz[3]**2*dpij[2][1]*dpij[2][0]*rk[3] - (dpij[2][1]*ek[3]-hk[3]*dpij[2][0])*(rk[3]+dz[3]**2/(rk[3] + core_size))*dpij[2][0]) / ((dz[2]*rk[3]*dpij[2][0])**2 + (dpij[2][1]*ek[3]-hk[3]*dpij[2][0])**2 + core_size) + # Term 6

            (2*dz[3]**2*dpij[3][1]*dpij[3][0]*rk[3] - (dpij[3][1]*ek[3]-hk[3]*dpij[3][0])*(rk[3]+dz[3]**2/(rk[3] + core_size))*dpij[3][0]) / ((dz[3]*rk[3]*dpij[3][0])**2 + (dpij[3][1]*ek[3]-hk[3]*dpij[3][0])**2 + core_size) - # Term 7
            (2*dz[0]**2*dpij[3][1]*dpij[3][0]*rk[0] - (dpij[3][1]*ek[0]-hk[0]*dpij[3][0])*(rk[0]+dz[0]**2/(rk[0] + core_size))*dpij[3][0]) / ((dz[3]*rk[0]*dpij[3][0])**2 + (dpij[3][1]*ek[0]-hk[0]*dpij[3][0])**2 + core_size) # Term 8
        )

        # NOTE: we can't set this to a csdl variable here because the shapes of the inputs change (like dx[0] based on the number of timesteps)
        # doublet_velocity = doublet_velocity.set(csdl.slice[], value=u)
        # doublet_velocity = doublet_velocity.set(csdl.slice[], value=v)
        # doublet_velocity = doublet_velocity.set(csdl.slice[], value=w)
        return u, v, w