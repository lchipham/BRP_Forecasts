""" Get bond price from YTM """

def bond_price(par, T, ytm, coup, freq = 2): #coup in %
    freq = float(freq)
    periods = T*freq
    coupon = coup/100.* par/freq
    dt = [(i + 1)/freq for i in range(int(periods))] #+1 as the first value of range is 0
    price = sum([coupon / (1+ytm/freq)**(freq*t) for t in dt]) + \
            par/(1+ytm/freq)**(freq*T)
    return price