"""
Convexity: sensitivity measure of the duration of a bond to yield changes.
Convexity is a risk management tool which measures the amount of market risk in a bond portfolio.
Higher convexity portfolios are less affected by interest rate volatility than lower convexity portfolio,
given the same bond duration and yield.
--> Higher convexity bonds are more expensive than lower convexity ones, all else equal.

Modified Duration = (Pdown + Pup - 2*Price) / (Price*(dY**2))
+ Pdown: price - 0.01
+ Pup: price + 0.01
+ Price: initial price of bond
+ dY: change in yield
"""
from bond_price import bond_price
from bond_ytm import bond_YTM

def bond_convexity(price, par, T, coup, freq, dY = 0.01):
    ytm = bond_YTM(price, par, T, coup, freq)
    ytm_minus = ytm - dY
    ytm_plus = ytm + dY
    p_down = bond_price(par, T, ytm_minus, coup, freq)
    p_up = bond_price(par, T, ytm_plus, coup, freq)
    convexity = (p_down + p_up - 2*price) / (price*(dY**2))
    return convexity