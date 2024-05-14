import numpy as np 
import csdl_alpha as csdl

def atan2_switch_python(x, y, scale=10000.):
    # # x > 0
    # f1 = np.arctan(y/x) * (0.5*np.tanh(scale*(x-1.e-2)) + 0.5)
    # # x < 0
    # f2 = (np.arctan(y/x) + np.pi) * (0.5*np.tanh(scale*(-1.e-2-x)) + 0.5)*(0.5*np.tanh(scale*(y)) + 0.5) # y >= 0
    # f3 = (np.arctan(y/x) - np.pi) * (0.5*np.tanh(scale*(-1.e-2-x)) + 0.5)*(0.5*np.tanh(scale*(-y)) + 0.5) # y < 0
    # # x = 0
    # f4 = np.pi/2 * (0.5*(np.tanh(scale*(x+1e-2)) - np.tanh(scale*(x-1e-2)))) * (0.5*np.tanh(scale*(y+1e-3)) + 0.5) # y >= 0
    # f5 = -np.pi/2 * (0.5*(np.tanh(scale*(x+1e-2)) - np.tanh(scale*(x-1e-2)))) * (0.5*np.tanh(scale*(1e-3-y)) + 0.5) # y < 0

    # x > 0
    f1 = np.arctan(y/x) * (0.5*np.tanh(scale*(x*.99)) + 0.5)
    # x < 0
    f2 = (np.arctan(y/x) + np.pi) * (0.5*np.tanh(scale*(-1.01*x)) + 0.5)*(0.5*np.tanh(scale*(y)) + 0.5) # y >= 0
    f3 = (np.arctan(y/x) - np.pi) * (0.5*np.tanh(scale*(-1.01*x)) + 0.5)*(0.5*np.tanh(scale*(-y)) + 0.5) # y < 0
    # x = 0
    f4 = np.pi/2 * (0.5*(np.tanh(scale*(x*1.05)) - np.tanh(scale*(x*.95)))) * (0.5*np.tanh(scale*(y*1.05)) + 0.5) # y >= 0
    f5 = -np.pi/2 * (0.5*(np.tanh(scale*(x*1.05)) - np.tanh(scale*(x*.95)))) * (0.5*np.tanh(scale*(-.95*y)) + 0.5) # y < 0

    # return f1 + f2 + f3
    return f1 + f2 + f3 + f4 + f5

def atan2_switch(x, y, scale=10000.):
    # # x > 0
    # f1 = np.arctan(y/x) * (0.5*np.tanh(scale*(x-1.e-2)) + 0.5)
    # # x < 0
    # f2 = (np.arctan(y/x) + np.pi) * (0.5*np.tanh(scale*(-1.e-2-x)) + 0.5)*(0.5*np.tanh(scale*(y)) + 0.5) # y >= 0
    # f3 = (np.arctan(y/x) - np.pi) * (0.5*np.tanh(scale*(-1.e-2-x)) + 0.5)*(0.5*np.tanh(scale*(-y)) + 0.5) # y < 0
    # # x = 0
    # f4 = np.pi/2 * (0.5*(np.tanh(scale*(x+1e-2)) - np.tanh(scale*(x-1e-2)))) * (0.5*np.tanh(scale*(y+1e-3)) + 0.5) # y >= 0
    # f5 = -np.pi/2 * (0.5*(np.tanh(scale*(x+1e-2)) - np.tanh(scale*(x-1e-2)))) * (0.5*np.tanh(scale*(1e-3-y)) + 0.5) # y < 0

    # x > 0
    f1 = csdl.arctan(y/x) * (0.5*custom_tanh(scale*(x*.99)) + 0.5)
    # x < 0
    f2 = (csdl.arctan(y/x) + np.pi) * (0.5*custom_tanh(scale*(-1.01*x)) + 0.5)*(0.5*custom_tanh(scale*(y)) + 0.5) # y >= 0
    f3 = (csdl.arctan(y/x) - np.pi) * (0.5*custom_tanh(scale*(-1.01*x)) + 0.5)*(0.5*custom_tanh(scale*(-y)) + 0.5) # y < 0
    # x = 0
    f4 = np.pi/2 * (0.5*(custom_tanh(scale*(x*1.05)) - custom_tanh(scale*(x*.95)))) * (0.5*custom_tanh(scale*(y*1.05)) + 0.5) # y >= 0
    f5 = -np.pi/2 * (0.5*(custom_tanh(scale*(x*1.05)) - custom_tanh(scale*(x*.95)))) * (0.5*custom_tanh(scale*(-.95*y)) + 0.5) # y < 0

    # return f1 + f2 + f3
    return f1 + f2 + f3 + f4 + f5

def custom_tanh(x):
    tanh = (csdl.exp(x) - csdl.exp(-x)) / (csdl.exp(x) + csdl.exp(-x))
    return tanh