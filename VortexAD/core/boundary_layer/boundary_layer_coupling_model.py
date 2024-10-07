import numpy as np
import csdl_alpha as csdl
from VortexAD.utils.csdl_switch import switch_func
import ozone_alpha as ozone

def boundary_layer_coupling_model(Ue, BL_mesh, dx, nu):

    nc_BL = BL_mesh.shape[1]
    ns = BL_mesh.shape[2] + 1

    dUe_dx = csdl.Variable(value=np.zeros(Ue.shape))
    dUe_dx = dUe_dx.set(csdl.slice[:,1:-1,:], value=(Ue[:,2:,:] - Ue[:,:-2,:])/(dx[:,1:,:] + dx[:,:-1,:])) # central difference
    dUe_dx = dUe_dx.set(csdl.slice[:,0,:], value=(-3*Ue[:,0,:] + 4*Ue[:,1,:] - Ue[:,2,:]) / (dx[:,0,:] + dx[:,1,:])) # backward difference
    dUe_dx = dUe_dx.set(csdl.slice[:,-1,:], value=(Ue[:,-3,:] - 4*Ue[:,-2,:] + 3*Ue[:,-1,:]) / (dx[:,-2,:] + dx[:,-1,:])) # forward difference

    vel_integrand = (Ue[:,:-1,:]**5 + Ue[:,1:,:]**5)/2*dx

    vel_integration = csdl.Variable(value=np.zeros(Ue.shape))
    dx_grid = csdl.Variable(value=np.zeros(Ue.shape))
    for ind in range(1,nc_BL):
        dx_grid = dx_grid.set(
            csdl.slice[:,ind,:],
            value=csdl.sum(dx[:,:ind,:], axes=(1,))
        )
        vel_integration = vel_integration.set(
            csdl.slice[:,ind,:],
            value=csdl.sum(vel_integrand[:,:ind,:], axes=(1,))
        )
    theta_lam = (0.45*nu/Ue**6 * vel_integration)**0.5

    lam = theta_lam**2/nu * dUe_dx

    H_func_list = [
        0.0731/(0.14 + lam) + 2.088,
        2.61 - 3.75*lam + 5.24*lam**2,
    ]
    H_lam = switch_func(lam, H_func_list, [0], scale=100.)

    l_func_list = [
        0.22 +1.402*lam + 0.018*lam/(0.107+lam),
        0.22+1.57*lam-1.8*lam**2
    ]
    l = switch_func(lam, l_func_list, [0], scale=100.)

    Cf_lam = 2*nu/Ue/theta_lam * l
    delta_star_lam = H_lam*theta_lam

    Re_theta = Ue*theta_lam/nu
    Re_x = Ue*dx_grid/nu
    Michel_criterion = 1.174*(1+(22400/Re_x))*Re_x**0.46

    delta_MC = Re_theta-Michel_criterion

    delta_MC_val = delta_MC.value
    sep_ind_span = []
    # for s in range(ns-1):
    #     try:
    #         ind = np.where(delta_MC_val[:,:,s] > 0)[1][0]
    #     except:
    #         ind = nc_BL
    #     sep_ind_span.append(ind)
    # sep_ind = sep_ind_span[0]

    sep_ind = 8 # aoa=10
    # sep_ind = 22 # aoa=6
    # sep_ind = 46 # aoa=0, 21 chordwise nodes with uneven spacing
    sep_ind = 67 # aoa=0, 31 chordwise with even spacing

    # Ozone integration for turbulent boundary layer
    # NOTE: we are only doing this for time ind -2 to save cost
    # NOTE: we need to reshape everything to (num_c_nodes, ns)
    
    # reshape step
    target_shape = (nc_BL-sep_ind, ns-1)
    Ue_reshaped = Ue[:,sep_ind:,:].reshape(target_shape)
    dUe_dx_reshaped = dUe_dx[:,sep_ind:,:].reshape(target_shape)

    # compute initial state values here
    H0 = H_lam[:,sep_ind,:].reshape((ns-1,))
    H0 = 1.4
    theta_0_ode = theta_lam[:,sep_ind,:].reshape((ns-1,))

    H1_0 = 3.0445 + 0.8702/(H0-1.1)**1.271
    Ue_H1_theta_0_ode = Ue_reshaped[0,:]*H1_0*theta_0_ode

    approach = ozone.approaches.TimeMarching()
    ode_problem = ozone.ODEProblem('RK4', approach)
    ode_problem.add_state('theta', theta_0_ode, store_history=True)
    ode_problem.add_state('Ue_H1_theta', Ue_H1_theta_0_ode)
    ode_problem.add_dynamic_parameter('Ue', Ue_reshaped)
    ode_problem.add_dynamic_parameter('dUe_dx', dUe_dx_reshaped)
    ode_problem.set_timespan(
        ozone.timespans.StepVector(start=0, step_vector=dx[0,sep_ind:,int((ns-1)/2)])
    )
    ode_problem.set_function(
        turbulent_BL_ode_function,
        nu=nu
    )
    ode_outputs = ode_problem.solve()

    delta_star_turb = ode_outputs.profile_outputs['delta_star']
    H_turb = ode_outputs.profile_outputs['H']
    Cf_turb = ode_outputs.profile_outputs['Cf'] 
    theta_turb = ode_outputs.states['theta']

    delta_star = csdl.Variable(value=np.zeros(delta_star_lam.shape))
    delta_star = delta_star.set(csdl.slice[:,:sep_ind,:], value=delta_star_lam[:,:sep_ind,:])
    delta_star = delta_star.set(csdl.slice[:,sep_ind:,:], value=delta_star_turb.reshape(1,nc_BL-sep_ind,ns-1))
    theta = csdl.Variable(value=np.zeros(delta_star_lam.shape))
    theta = theta.set(csdl.slice[:,:sep_ind,:], value=theta_lam[:,:sep_ind,:])
    theta = theta.set(csdl.slice[:,sep_ind:,:], value=theta_turb.reshape(1,nc_BL-sep_ind,ns-1))
    H = csdl.Variable(value=np.zeros(delta_star_lam.shape))
    H = H.set(csdl.slice[:,:sep_ind,:], value=H_lam[:,:sep_ind,:])
    H = H.set(csdl.slice[:,sep_ind:,:], value=H_turb.reshape(1,nc_BL-sep_ind,ns-1))
    Cf = csdl.Variable(value=np.zeros(delta_star_lam.shape))
    Cf = Cf.set(csdl.slice[:,:sep_ind,:], value=Cf_lam[:,:sep_ind,:])
    Cf = Cf.set(csdl.slice[:,sep_ind:,:], value=Cf_turb.reshape(1,nc_BL-sep_ind,ns-1))

    return delta_star, theta, H, Cf

def turbulent_BL_ode_function(ozone_vars:ozone.FuncVars,nu):
    theta = ozone_vars.states['theta']
    Ue_H1_theta = ozone_vars.states['Ue_H1_theta']

    Ue = ozone_vars.dynamic_parameters['Ue']
    dUe_dx = ozone_vars.dynamic_parameters['dUe_dx']

    H1 = Ue_H1_theta/(Ue*theta)
    # H = ((H1-3.0445)/.8702)**(-1/1.2721)

    func1 = ((H1-3.3)/.8234)**(-1/1.287) + 1.1
    func2 = ((H1-3.3)/1.5501)**(-1/3.064) + .6778
    H = switch_func(H1, [func1, func2], [5.3], scale=100.)
    Re_theta = Ue*theta/nu
    Cf = 0.246*csdl.power(10., -.678*H)*Re_theta**(-.268)
    disp_thickness = H*theta

    dtheta_dx = Cf/2 - (H+2)*theta/Ue*dUe_dx

    F = 0.0306*(H1-3.0)**(-.6169)
    dUeH1theta_dx = Ue*F

    ozone_vars.d_states['theta'] = dtheta_dx
    ozone_vars.d_states['Ue_H1_theta'] = dUeH1theta_dx

    ozone_vars.profile_outputs['delta_star'] = disp_thickness
    ozone_vars.profile_outputs['Cf'] = Cf
    ozone_vars.profile_outputs['H'] = H
