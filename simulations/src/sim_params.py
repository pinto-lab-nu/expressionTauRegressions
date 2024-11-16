
### Calcium Dynamic Parameters ###
Ca_rest = 50e-9
Ca_extrusion = 1800
Ca_binding_ratio = 110
Ca_affinity = 144e-9 #Chen et al 2013 %Dana et al 2019 >> 147e-9
indicator_conc = 200e-6
#dt = 1/100
spike_amp = 7.6e-6 / 100 # modify the spike amplitude (Ca concentration, M, from A. Song et al. 2021)

Ca_trans_amp = 0.09
tau_on = 0.115 # Dana et al 2019
tau_off = 1.525 # Dana et al 2019

dynamic_range = 53.8 # Dana et al 2019
K_D = 147e-9
hill_coefficient = 2.90 # Chen et al 2013
F0 = 1

dt = 0.001 # time interval, in seconds, of simulation
t_start = 0 # in seconds
t_end = 480 # in seconds

