import numpy as np 
import matplotlib.pyplot as plt 

ns_5 = {
    'nc': [9, 21, 31, 41, 51, 61, 71, 81],
    't_run': np.array([0.05300283432, 0.07506918907, 0.1083004475, 0.1594355106, 0.2039513588, 0.2771751881, 0.3362309933, 0.4615471363]),
    't_deriv': np.array([0.0932135582, 0.1549673080444336, 0.2592692375, 0.415627718, 0.5720000267, 0.8193700314, 0.9085962772, 1.155331135])
}

ns_11 = {
    'nc': [9, 21, 31, 41, 51, 61, 71, 81],
    't_run': np.array([0.08147001266, 0.207583189, 0.3863985538, 0.6798670292, 1.043791533, 1.401135206, 1.853725433, 2.10636258125305]),
    't_deriv': np.array([0.1605653763, 0.5988249779, 1.050513268, 1.83328414, 2.820245981, 3.783567429, 5.079591036, 6.750824451])
}

ns_21 = {
    'nc': [9, 21, 31, 41, 51, 61, 71, 81],
    't_run': np.array([0.1619672775, 0.6790888309, 1.392950535, 2.243501663, 3.353945732, 4.875445127, 6.423438072, 9.027449369]),
    't_deriv': np.array([0.4101006985, 1.789738655, 3.752719879, 6.979291916, 10.47275424, 15.14620352, 20.44470024, 27.64061594])
}

ns_31 = {
    'nc': [9, 11, 21, 31, 41, 51],
    't_run': np.array([0.2977676392, 0.4109201431, 1.487539768, 2.72794342, 4.923594952, 7.883049488]),
    't_deriv': np.array([0.8237714767, 1.073438644, 3.85003829, 8.655464888, 15.13029361, 24.43187284])
}

ns_41 = {
    'nc': [9, 11, 21, 31, 41],
    't_run': np.array([0.4732012749, 0.7233281136, 2.315705299, 4.841632128, 9.259945869]),
    't_deriv': np.array([1.2967484, 1.834450722, 7.014920712, 15.40431046, 27.8374486])
}

ns_dict = {
    '5': ns_5,
    '11': ns_11,
    '21': ns_21,
    '31': ns_31,
    '41': ns_41,
}

fig, (ax1, ax2) = plt.subplots(2,1,sharex=True)

nc_ticks = np.array([5, 9, 21, 31, 41, 51, 61, 71, 81])
t_run_ticks = [1, 5, 10, 20, 40]
t_deriv_ticks = [1, 5, 10, 25, 50, 100]



for ns in ns_dict.keys():
    ax1.plot(ns_dict[ns]['nc'], ns_dict[ns]['t_run'], '-*', label=f'ns = {ns}')
    ax2.plot(ns_dict[ns]['nc'], ns_dict[ns]['t_deriv'], '-*', label=f'ns = {ns}')

ax1.plot(nc_ticks, nc_ticks**2/nc_ticks[0]**4, '--k', label='Quadratic scaling')
ax2.plot(nc_ticks, nc_ticks**2/nc_ticks[0]**4, '--k', label='Quadratic scaling')


ax1.set_ylabel('Forward eval time (seconds)', fontsize=15)
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.grid(True)
ax1.yaxis.set_ticks(t_run_ticks)
ax1.set_yticklabels(t_run_ticks, fontsize=15)
ax1.xaxis.set_ticks(nc_ticks)
ax1.set_xticklabels(nc_ticks, fontsize=15)
# ax1.set_xlabel('Number of chordwise nodes (one-way)', fontsize=15)
# ax1.legend(fontsize=15)

ax2.set_ylabel('Derivative time (seconds)', fontsize=15)
ax2.set_xlabel('Number of chordwise nodes', fontsize=15)
ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.grid(True)
ax2.yaxis.set_ticks(t_deriv_ticks)
ax2.set_yticklabels(t_deriv_ticks, fontsize=15)
ax2.xaxis.set_ticks(nc_ticks)
ax2.set_xticklabels(nc_ticks, fontsize=15)

ax2.legend(fontsize=12)

fig = plt.figure()
for ns in ns_dict.keys():
    plt.plot(ns_dict[ns]['nc'], ns_dict[ns]['t_deriv'] / ns_dict[ns]['t_run'], '-*', label=f'ns = {ns}')
plt.plot(nc_ticks, 3*np.ones_like(nc_ticks), '--k', label='3x scaling')

plt.grid(True)
plt.legend(fontsize=12)
plt.ylim([0, 4])
plt.xlabel('Number of chordwise nodes', fontsize=15)
plt.ylabel('Derivative vs. forward run time ratio', fontsize=15)

plt.show()