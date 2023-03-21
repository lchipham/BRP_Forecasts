"""
Modified Duration: measures the % change in bond price with respect to a % change in yield.
(typically 1 percent or 100 basis points (bps)).
The higher the duration of a bond, the more sensitive it is to yield changes.

Modified Duration = (Pdown - Pup) / (2*Price*dY)
+ Pdown: price - 0.01
+ Pup: price + 0.01
+ Price: initial price of bond
+ dY: change in yield

** NOTE:
Duration describes the linear price-yield relationship for a small change in Y.
Because the yield curve is not linear, using a large value of dy does not approximate the duration measure well.
"""
from bond_price import bond_price
from bond_ytm import bond_YTM

def bond_moddur(price, par, T, coup, freq, dY = 0.01):
    ytm = bond_YTM(price, par, T, coup, freq)
    ytm_minus = ytm - dY
    ytm_plus = ytm + dY
    p_down = bond_price(par, T, ytm_minus, coup, freq)
    p_up = bond_price(par, T, ytm_plus, coup, freq)
    mod_duration = (p_down - p_up) / (2*price*dY)
    return mod_duration

