import csdl_alpha as csdl 
import numpy as np 

from VortexAD.utils.csdl_switch import switch_func

import ozone_alpha as ozone

def boundary_layer_solver(mesh_dict, output_dict, boundary_layer, num_nodes, nt, dt):
    surface_names = list(mesh_dict.keys())
    start, stop = 0, 0

    BL_output_dict = {}
    for i in range(len(surface_names)):

        surface_name = surface_names[i]
        nc, ns = mesh_dict[surface_name]['nc'], mesh_dict[surface_name]['ns']
        body_vel = output_dict[surface_name]['body_vel']
        mesh = mesh_dict[surface_name]['mesh']

        LE_ind = int((nc-1)/2)

        body_vel_ss = body_vel[:,:,LE_ind:,:]
        mesh_ss = mesh[:,:,LE_ind:,:,:]

        num_BL_vel_chord = int((nc+1)/2)
        # num_BL_vel_chord = nc
        BL_vel = csdl.Variable(value=np.zeros((num_nodes, nt, num_BL_vel_chord, ns-1))) # VELOCITY NORM
        BL_vel = BL_vel.set(csdl.slice[:,:,1:-1,:], value=(body_vel_ss[:,:,1:,:]+body_vel_ss[:,:,:-1,:])/2)
        BL_vel = BL_vel.set(csdl.slice[:,:,0,:], value=(body_vel[:,:,LE_ind-1,:]+body_vel[:,:,LE_ind,:])/2)
        BL_vel = BL_vel.set(csdl.slice[:,:,-1,:], value=(body_vel[:,:,-1,:]+body_vel[:,:,0,:])/2)

        '''
        BL_vel holds the boundary layer velocity at the mesh panel edges, interpolated from the collocation points.
        We need this to interpolate the edge velocities at the boundary layer mesh nodes
        Ue (below) holds the edge velocity on the actual boundary layer mesh 
        '''

        BL_mesh = boundary_layer[i] # for now this only holds the mesh
        nc_BL = BL_mesh.shape[2]
        nc_one_way = int((nc+1)/2)
        BL_per_panel = int((nc_BL-1)/(nc_one_way-1))

        BL_mesh = (BL_mesh[:,:,:,:-1,:]+BL_mesh[:,:,:,1:,:])/2. # turning from ns to ns-1
        dx = csdl.norm(BL_mesh[:,:,1:,:,:] - BL_mesh[:,:,:-1,:,:], axes=(4,))

        mesh_ss_span_center = (mesh_ss[:,:,:,:-1,:]+mesh_ss[:,:,:,1:,:])/2. # turning from ns to ns-1

        BL_vel_slope_panel = (BL_vel[:,:,1:,:] - BL_vel[:,:,:-1,:])/csdl.norm(mesh_ss_span_center[:,:,1:,:,:] - mesh_ss_span_center[:,:,:-1,:,:], axes=(4,))

        nu = 1.52e-5
        Ue = csdl.Variable(value=np.zeros((num_nodes, nt, nc_BL, ns-1))) # edge velocity norm
        Ue = Ue.set(csdl.slice[:,:,::BL_per_panel,:], value=BL_vel)

        for ind in range(BL_per_panel): # looping through num BL points per panel
            interp_value = BL_vel[:,:,:-1,:] + BL_vel_slope_panel*csdl.norm(BL_mesh[:,:,ind:-1:BL_per_panel,:,:]-mesh_ss_span_center[:,:,:-1,:,:], axes=(4,))
            Ue = Ue.set(csdl.slice[:,:,ind:-1:BL_per_panel,:], value=interp_value)

        dUe_dx = csdl.Variable(value=np.zeros(Ue.shape))
        dUe_dx = dUe_dx.set(csdl.slice[:,:,1:-1,:], value=(Ue[:,:,2:,:] - Ue[:,:,:-2,:])/(dx[:,:,1:,:] + dx[:,:,:-1,:])) # central difference
        dUe_dx = dUe_dx.set(csdl.slice[:,:,0,:], value=(-3*Ue[:,:,0,:] + 4*Ue[:,:,1,:] - Ue[:,:,2,:]) / (dx[:,:,0,:] + dx[:,:,1,:])) # backward difference
        dUe_dx = dUe_dx.set(csdl.slice[:,:,-1,:], value=(Ue[:,:,-3,:] - 4*Ue[:,:,-2,:] + 3*Ue[:,:,-1,:]) / (dx[:,:,-2,:] + dx[:,:,-1,:])) # forward difference

        vel_integrand = (Ue[:,:,:-1,:]**5 + Ue[:,:,1:,:]**5)/2*dx

        vel_integration = csdl.Variable(value=np.zeros(Ue.shape))
        dx_grid = csdl.Variable(value=np.zeros(Ue.shape))
        for ind in range(1,nc_BL):
            dx_grid = dx_grid.set(
                csdl.slice[:,:,ind,:],
                value=csdl.sum(dx[:,:,:ind,:], axes=(2,))
            )
            vel_integration = vel_integration.set(
                csdl.slice[:,:,ind,:],
                value=csdl.sum(vel_integrand[:,:,:ind,:], axes=(2,))
            )
        theta = (0.45*nu/Ue**6 * vel_integration)**0.5

        # theta_0 = (0.075/dUe_dx[:,:,0,:])**0.5
        # theta = (0.45*nu/Ue**6 * vel_integration)**0.5 + csdl.expand(theta_0, Ue.shape, 'ijk->ijak')
        # NOTE: SET THE INITIAL FINITE MOMENTUM THICKNESS HERE TO AVOID NAN
        # theta = theta.set(csdl.slice[:,:,0,:], value=(0.075/dUe_dx[:,:,0,:])**0.5)

        lam = theta**2/nu * dUe_dx

        H_func_list = [
            0.0731/(0.14 + lam) + 2.088,
            2.61 - 3.75*lam + 5.24*lam**2,
        ]
        H = switch_func(lam, H_func_list, [0], scale=100.)

        l_func_list = [
            0.22 +1.402*lam + 0.018*lam/(0.107+lam),
            0.22+1.57*lam-1.8*lam**2
        ]
        l = switch_func(lam, l_func_list, [0], scale=100.)

        C_f = 2*nu/Ue/theta * l
        disp_thickness = H*theta

        Re_theta = Ue*theta/nu
        Re_x = Ue*dx_grid/nu
        Michel_criterion = 1.174*(1+(22400/Re_x))*Re_x**0.46

        delta_MC = Re_theta-Michel_criterion

    centerline_ind = int((ns-1)/2)

    delta_MC_val = delta_MC.value[:,-2,:,:]
    sep_ind_span = []
    for s in range(ns-1):
        ind = np.where(delta_MC_val[:,:,s] > 0)[1][0]
        sep_ind_span.append(ind)
    sep_ind = sep_ind_span[0]

    # separation index based on lambda
    # asdf = lam.value[0,-2,:,centerline_ind]
    # aaa = np.where(asdf < -0.0842)
    # sep_ind = aaa[0][0]

    # Ozone integration for turbulent boundary layer
    # NOTE: we are only doing this for time ind -2 to save cost
    # NOTE: we need to reshape everything to (num_c_nodes, ns)
    
    # reshape step
    target_shape = (nc_BL-sep_ind, ns-1)
    Ue_reshaped = Ue[:,-2,sep_ind:,:].reshape(target_shape)
    dUe_dx_reshaped = dUe_dx[:,-2,sep_ind:,:].reshape(target_shape)

    # compute initial state values here
    H0 = H[:,-2,sep_ind,:].reshape((ns-1,))
    H0 = 1.4
    theta_0_ode = theta[:,-2,sep_ind,:].reshape((ns-1,))

    H1_0 = 3.0445 + 0.8702/(H0-1.1)**1.271
    Ue_H1_theta_0_ode = Ue_reshaped[0,:]*H1_0*theta_0_ode

    approach = ozone.approaches.TimeMarching()
    ode_problem = ozone.ODEProblem('RK4', approach)
    ode_problem.add_state('theta', theta_0_ode, store_history=True)
    ode_problem.add_state('Ue_H1_theta', Ue_H1_theta_0_ode)
    ode_problem.add_dynamic_parameter('Ue', Ue_reshaped)
    ode_problem.add_dynamic_parameter('dUe_dx', dUe_dx_reshaped)
    ode_problem.set_timespan(
        ozone.timespans.StepVector(start=0, step_vector=dx[0,-2,sep_ind:,int((ns-1)/2)])
    )
    ode_problem.set_function(
        turbulent_BL_ode_function,
        nu=nu
    )
    ode_outputs = ode_problem.solve()

    disp_thickness_turb = ode_outputs.profile_outputs['delta_star']
    H_turb = ode_outputs.profile_outputs['H']
    Cf_turb = ode_outputs.profile_outputs['Cf'] 
    theta_turb = ode_outputs.states['theta']

    # plotting for laminar region ONLY
    BL_mesh_centerline = BL_mesh.value[0,0,:,centerline_ind,0] /BL_mesh.value[0,0,-1,centerline_ind,0]
    C_f_centerline = C_f.value[0,-2,:,centerline_ind]
    delta_star_centerline = disp_thickness.value[0,-2,:,centerline_ind]

    import matplotlib.pyplot as plt
    # plt.figure()
    # plt.plot(BL_mesh_centerline, C_f_centerline, '-^', label='Cf')
    # plt.plot(BL_mesh_centerline, delta_star_centerline, '-*', label='disp. thickness')
    # plt.plot(BL_mesh_centerline, theta[0,-2,:,centerline_ind].value, '-<', label='mom. thickness')
    # plt.ylim([-.002, 0.006])
    # plt.grid()
    # plt.legend()

    # plt.figure()
    # plt.plot(BL_mesh_centerline, delta_MC[0,-2,:,centerline_ind].value, '-^', label='Re_theta - Re_x')
    # # plt.plot(BL_mesh_centerline, delta_star_centerline, '-*', label='disp. thickness')
    # # plt.ylim([-.002, 0.006])
    # plt.grid()
    # plt.legend()

    # plt.figure()
    # plt.plot(BL_mesh_centerline, H[0,-2,:,centerline_ind].value, '-^', label='H')
    # # plt.plot(BL_mesh_centerline, delta_star_centerline, '-*', label='disp. thickness')
    # # plt.ylim([-.002, 0.006])
    # plt.grid()
    # plt.legend()

    C_f_centerline_turb = np.zeros(C_f_centerline.shape)
    C_f_centerline_turb[:sep_ind] = C_f_centerline[:sep_ind]
    C_f_centerline_turb[sep_ind:] = Cf_turb.value[:,centerline_ind]

    delta_star_turb = np.zeros(delta_star_centerline.shape)
    delta_star_turb[:sep_ind] = delta_star_centerline[:sep_ind]
    delta_star_turb[sep_ind:] = disp_thickness_turb.value[:,centerline_ind]

    theta_turb_centerline = np.zeros(C_f_centerline.shape)
    theta_turb_centerline[:sep_ind] = theta[0,-2,:sep_ind,centerline_ind].value
    theta_turb_centerline[sep_ind:] = theta_turb[:,centerline_ind].value

    plt.figure()
    plt.plot(BL_mesh_centerline, C_f_centerline_turb, '-^', label='Cf')
    plt.plot(BL_mesh_centerline, delta_star_turb, '-*', label='displacement thickness (m)')
    # plt.plot(BL_mesh_centerline, theta_turb_centerline, '-<', label='mom. thickness')
    plt.ylim([-.002, 0.006])
    plt.xlabel('x/c')
    plt.grid()
    plt.legend()

    H_turb_centerline = np.zeros(C_f_centerline.shape)
    H_turb_centerline[:sep_ind] = H[0,-2,:sep_ind,centerline_ind].value
    H_turb_centerline[sep_ind:] = H_turb[:,centerline_ind].value

    plt.figure()
    plt.plot(BL_mesh_centerline, H_turb_centerline, '-^', label='H')
    # plt.plot(BL_mesh_centerline, delta_star_centerline, '-*', label='disp. thickness')
    # plt.ylim([-.002, 0.006])
    plt.grid()
    plt.legend()


    plt.show()

    pass

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
