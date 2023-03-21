"""
The following code generates a list of forward rates from a list of spot rates. (from the second input onward)
"""
class ForwardRates():
    def __init__(self):
        self.forward_rates = [] #list
        self.spot_rates = dict() #dict

    def add_spot_rate(self, T, spot):
        self.spot_rates[T] = spot

    def _calc_forward_rate_(self, T1, T2):
        r1 = self.spot_rates[T1]
        r2 = self.spot_rates[T2]
        forward_rate = round((r2*T2 - r1*T1) / (T2 - T1), 4)
        return forward_rate

    def get_forward_rates(self):
        """
        :return: a list of forward rates, starting from the next time period.
        """
        sorted_Ts = sorted(self.spot_rates.keys())
        for T1, T2 in zip(sorted_Ts, sorted_Ts[1:]):
            forward_rate = self._calc_forward_rate_(T1, T2)
            self.forward_rates.append(forward_rate)

        return self.forward_rates



