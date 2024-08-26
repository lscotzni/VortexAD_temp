import numpy as np 
import matplotlib.pyplot as plt 

ns_5 = {
    'nc': [5, 11, 16, 21, 31, 41],
    't_run': np.array([2.424628258, 2.59408474, 2.69256258, 2.782936811, 2.906484127, 3.536058187]),
    't_deriv': np.array([9.942940712, 10.57206726, 10.98830128, 11.94970679, 12.89677501, 14.69954038]),
    'memory': np.array([599.2695313, 1305.054688, 2311.878906, 2973.859375, 2542.609375, 4179.195313]) / 1000.
}



ns_11 = {
    'nc': [5, 11, 16, 21, 31, 41],
    't_run': np.array([2.65329957, 3.169378042, 3.589165688, 5.018176079, 6.275646687, 7.340821266]),
    't_deriv': np.array([11.11500788, 13.56399703, 15.84024596, 21.07227325, 26.21526885, 37.26783562]),
    'memory': np.array([1530.109375, 2325.890625, 4108.933594, 6270.507813, 13053.10156, 22772.16406]) / 1000.
}

ns_21 = {
    'nc': [5, 11, 16, 21, 31, 41],
    't_run': np.array([3.70062995, 4.822455168, 5.587801456, 9.743909597, 13.63217092, 18.64187407]),
    't_deriv': np.array([15.60070372, 21.2190218, 30.57040858, 41.24806595, 70.25166392, 114.4406204]),
    'memory': np.array([3904.039063, 7349.886719, 14091.95313, 23106.91797, 50657.78906, 89518.10938]) / 1000.
}

ns_31 = {
    'nc': [5, 11, 16, 21, 26],
    't_run': np.array([3.773807764, 8.944212198, 11.6991303, 12.80703473, 16.7565124]),
    't_deriv': np.array([20.4165411, 33.59921598, 53.91567755, 75.53060794, 108.609406]),
    'memory': np.array([4442.796875, 15796.43359, 30967.44141, 51190.22266, 76954.41016]) / 1000.
}

ns_41 = {
    'nc': [5, 11, 16],
    't_run': np.array([5.072773695, 9.136147499, 12.93366766]),
    't_deriv': np.array([21.02411151, 45.74886012, 79.98131442]),
    'memory': np.array([7406.980469, 27539.08594, 52856.76172]) / 1000.
}

ns_dict = {
    '5': ns_5,
    '11': ns_11,
    '21': ns_21,
    '31': ns_31,
    '41': ns_41,
}

fig, (ax1, ax2) = plt.subplots(2,1,sharex=True)

nc_ticks = np.array([5, 11, 16, 21, 31, 41])
t_run_ticks = [1, 5, 10, 20, 40]
t_deriv_ticks = [1, 5, 10, 25, 50, 100]



for ns in ns_dict.keys():
    ax1.plot(ns_dict[ns]['nc'], ns_dict[ns]['t_run'], '-*', label=f'ns = {ns}')
    ax2.plot(ns_dict[ns]['nc'], ns_dict[ns]['t_deriv'], '-*', label=f'ns = {ns}')

ax1.plot(nc_ticks, nc_ticks**2/5, '--k', label='Quadratic scaling')
ax2.plot(nc_ticks, (nc_ticks)**2/2.5, '--k', label='Quadratic scaling')


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
plt.plot(nc_ticks, 5*np.ones_like(nc_ticks), '--k', label='5x scaling')

plt.grid(True)
plt.legend(fontsize=12)
# plt.ylim([0, 4])
plt.xlabel('Number of chordwise nodes', fontsize=15)
plt.ylabel('Adjoint run time vs. forward run time ratio', fontsize=15)

plt.show()