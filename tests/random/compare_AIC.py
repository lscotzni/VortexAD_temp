import numpy as np 
import pickle

nc, ns = 10, 4
nt = 10

vlm_AIC_file = open('vlm_AIC', 'rb')
vlm_AIC = pickle.load(vlm_AIC_file)
vlm_AIC_file.close()

vlm_KC_file = open('vlm_kutta_condition', 'rb')
vlm_KC = pickle.load(vlm_KC_file)
vlm_KC_file.close()

vlm_gamma_file = open('vlm_gamma', 'rb')
vlm_gamma = pickle.load(vlm_gamma_file).reshape((nc,ns))
vlm_gamma_file.close()

pm_AIC_file = open('pm_AIC', 'rb')
pm_AIC = pickle.load(pm_AIC_file)
pm_AIC_file.close()

pm_AIC_wake_file = open('pm_AIC_wake', 'rb')
pm_AIC_wake = pickle.load(pm_AIC_wake_file)
pm_AIC_wake_file.close()

pm_KC_file = open('pm_kutta_condition', 'rb')
pm_KC = pickle.load(pm_KC_file)
pm_KC_file.close()

pm_gamma_file = open('pm_gamma', 'rb')
pm_gamma = pickle.load(pm_gamma_file).reshape((nt, nc,ns))
pm_gamma_file.close()