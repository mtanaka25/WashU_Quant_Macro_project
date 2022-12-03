# Default values
beta_def = 0.975 # Stokey (2009)'s exp(-r)
alpha_def = 0.745 #Stokey (2009)'s (1 - omega)
sigma_def = 0.5 #Stokey (2009)'s epsilon
gamma_def = 3.5 #Stokey (2009)'s theta
d_def = 0.2 # To replicate houseownership rate in the US
ph_def = 4. # To replicate houseownership rate in the US
#ph_def = 10. # To replicate houseownership rate in the US
h_star_def = 1.0 # To replicate houseownership rate in the US
#h_star_def = 0.8 # To replicate houseownership rate in the US
h_eps_def = 0.8 # To replicate houseownership rate in the US
#h_eps_def = 0.5 # To replicate houseownership rate in the US
c_d_def = 0.1 # To replicate houseownership rate in the US
c_hat_d_def = 999. # is not used so far
delta_def = 1.# we used in an assignment
a_d_max_def = 5.
lmbd_def = 0. # is not used so far
#lmbd_def = 0.4 # Stokey (2009)'s lambda(0.08) * h_star
r_def = 0.01 #Stokey (2009)
theta_def = 1.0 # we used in an assignment
kappaH_def = 0.05 # we used in an assignment
kappaN_def = 0.05 # we used in an assignment
a_range_def = (0., 25.)
N_a_def = 51
max_iter_def = 2_000
tol_def = 1E-5