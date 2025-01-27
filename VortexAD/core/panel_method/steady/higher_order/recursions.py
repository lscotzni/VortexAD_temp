import numpy as np
import csdl_alpha as csdl

# H integrals
def H_113(a_bar_list, l1_list, l2_list, c1_list, c2_list, h):
    ne = len(l1_list)
    H113 = 0.
    for i in range(ne):
        NUM = a_bar_list[i]*(
            l2_list[i]*c1_list[i] - l1_list[i]*c2_list[i]
        )
        DEN = c1_list[i]*c2_list[i] + \
            a_bar_list[i]**2*l1_list[i]*l2_list[i]
        H113 = H113 + csdl.arctan(
            NUM/DEN
        ) # NOTE: NEED TO USE ATAN2
    H113 = H113 / h
    return H113

# F integrals
def F_111(l1, l2, g):
    F111 = csdl.log(
        ((l1**2 + g**2)**0.5-l1)*\
        ((l2**2 + g**2)**0.5+l2)/g**2
    )
    return F111

def F_113(g, nu_eta, nu_xi, R1, R2, xi, eta, point):
    xi1, xi2 = xi[0], xi[1]
    eta1, eta2 = eta[0], eta[1]
    x, y = point[0], point[1]

    E211 = E_211(R1, R2, xi1, xi2, x)
    E121 = E_121(R1, R2, eta1, eta2, y)

    F113 = 1/g**2 * (-nu_eta*E211 + nu_xi*E121)
    return F113

def F_123(a_bar, nu_xi, F113, nu_eta, R1, R2):
    F123 = a_bar*nu_eta*F113 - nu_xi*E_111(R1, R2)
    return F123

def F_213(a_bar, nu_xi, F113, nu_eta, R1, R2):
    F213 = a_bar*nu_xi*F113 + nu_eta*E_111(R1, R2)
    return F213

# E integrals
# NOTE: these don't need to be called externally
# they are only used in the F integrals above
def E_111(R1, R2):
    E111 = 1/R2 - 1/R1
    return E111

def E_211(R1, R2, xi1, xi2, x):
    E211 = (xi2-x)/R2 - (xi1-x)/R1
    return E211

def E_121(R1, R2, eta1, eta2, y):
    E121 = (eta2-y)/R2 - (eta1-y)/R1
    return E121