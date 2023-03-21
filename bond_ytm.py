""" Get yield-to-maturity of a bond """
import scipy.optimize as optimize

def bond_YTM(price, par, T, coup, freq=2, guess=0.05):
    freq = freq
    periods = T * freq
    coupon = coup / 100. * par / freq
    dt = [(i + 1) / freq for i in range(int(periods))] #+1 as the first value of range is 0
    ytm_func = lambda y: \
        sum([coupon / (1 + y / freq) ** (freq * t) for t in dt]) + \
        par / (1 + y / freq) ** (freq * T) - price
    return optimize.newton(ytm_func, guess)
