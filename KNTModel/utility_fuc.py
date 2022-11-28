from numba import njit
from numba import f8

@njit(f8(f8, f8, f8, f8, f8))
def utility(c, h, alpha, sigma, gamma):
    nested_ces = (alpha * c**((sigma - 1)/sigma)
                  + (1 - alpha) * h**((sigma - 1)/sigma)
                  )**(sigma/(sigma - 1))
    return 1/(1 - gamma) * nested_ces**(1 - gamma)

@njit(f8(f8, f8, f8, f8, f8, f8))
def cons(z, a, r, a_prime, down_pay=0., mortgage_pay=0.):
    return z + (1 + r)*a - a_prime - down_pay - mortgage_pay
