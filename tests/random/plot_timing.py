import numpy as np
import matplotlib.pyplot as plt

panels = ['1000', '5000', '10000', '15000']
pre_proc = np.array([9.18E-05,   0.0001270294189,  0.0001776218414, 0.0001999378204])
assembly = np.array([8.28E-02, 1.916995859, 6.748494053, 15.06816492])
lin_sys = np.array([0.06976447105, 0.4176476955, 2.855184984, 7.666922331])

total = pre_proc+assembly+lin_sys
pre_proc_r = pre_proc/total
assembly_r = assembly/total
lin_sys_r = lin_sys/total

plt.bar(panels, pre_proc_r)
plt.bar(panels, assembly_r, bottom=pre_proc_r, color='r')
plt.bar(panels, lin_sys_r, bottom=assembly_r+pre_proc_r, color='b')

plt.xlabel('number of panels')
plt.ylabel('fraction of total computation time')
plt.title('relative assembly time (red) vs. linear system solve time (blue)')
plt.show()

1