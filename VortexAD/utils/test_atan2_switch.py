import numpy as np
from VortexAD.utils.atan2_switch import atan2_switch_python, atan2_switch
import csdl_alpha as csdl

n = 10000
radius = 1
t = np.linspace(0,2*np.pi,n)
x, y = np.cos(t)*radius, np.sin(t)*radius

theta_exp = np.arctan2(y,x) * 180/np.pi
theta_python = atan2_switch_python(x,y) * 180/np.pi

recorder = csdl.Recorder(inline=True)
recorder.start()
theta_csdl = atan2_switch(x,y, scale=100.) * 180/np.pi
recorder.stop()

import matplotlib.pyplot as plt
plt.figure()
plt.plot(theta_python, 'k*', label='Python atan2 switch')
plt.plot(theta_exp, 'c', label='Expected')
plt.plot(theta_csdl.value, 'r', label='CSDL atan2 switch')
plt.xlabel('index')
plt.ylabel('theta (degrees)')
plt.grid()
plt.legend()

plt.show()
