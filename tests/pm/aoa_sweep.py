import numpy as np
import matplotlib.pyplot as plt



def McCroskey_fit(alpha):
    CL = alpha*1.074902/10.
    return CL

Abbott_data = {
    'alpha': np.array([-10.1947, -8.14138, -6.25579, -5.22822, -4.19972, -1.96944, 0., 0.940006, 1.96944, 2.99515, 3.85131, 4.87888, 5.90831, 7.96346, 10.1891, 11.0471, 13.1088, 16.3759]),
    'CL': np.array([-1.06927, -0.827958, -0.638207, -0.526128, -0.422627, -0.215533, 0, 0.120611, 0.215533, 0.34477, 0.439599, 0.551678, 0.6466, 0.870758, 1.12074, 1.19842, 1.36252, 1.59591])
}

Ladson_data = {
    'alpha': np.array([-4.04, -2.14, -.05 , 2.05 , 4.04 , 6.09 , 8.30 , 10.12, 11.13, 12.12, 13.08, 14.22, 15.26]),
    'CL': np.array([-.4417, -.2385, -.0126,  .2125,  .4316,  .6546,  .8873, 1.0707, 1.1685, 1.2605, 1.3455, 1.4365, 1.5129])
}

code_data = {
    'alpha': np.array([-10, -7.5, -5, -2.5, 0, 2.5, 5, 7.5, 10, 12.5, 15]),
    'CL': np.array([-0.98285139, -0.74257443, -0.4976574, -0.24961491, 1.91E-14, 0.24961491, 0.4976574, 0.74257443, 0.98285139, 1.21703071, 1.44372966])
}

McCroskey_data = McCroskey_fit(code_data['alpha'])

plt.figure()
plt.plot(Abbott_data['alpha'], Abbott_data['CL'], 'sr', label='Abbott data')
plt.plot(Ladson_data['alpha'], Ladson_data['CL'], '>b', label='Ladson data')
plt.plot(code_data['alpha'], code_data['CL'], '-*k', label='CSDL panel code')
plt.xlabel('alpha (deg)')
plt.ylabel('CL')
plt.legend()
plt.grid()

plt.show()