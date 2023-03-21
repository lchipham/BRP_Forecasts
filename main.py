# Classes
from bond_price import bond_price
from bond_ytm import bond_YTM
from bond_duration import bond_moddur
from bond_convexity import bond_convexity
from ForwardRates import ForwardRates
from BootstrapYC import BootstrapYieldCurve

# Modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    option_list = ["Bond Price", "Bond Yield-To-Maturity", "Bond Duration", "Bond Convexity", "Forward Rates", "Bootstrap Yield Curve"]

    while True:
        print("Select one of the following options:\n")
        for i in range(len(option_list)):
            print(str(option_list.index(option_list[i]) + 1) + ") ", option_list[i])
        try:
            user_input = int(input("Enter option number: "))
        except ValueError:
            print("Sorry, that was not a valid number. Please try again!")
            print("\n")
            continue
        print("---------------------", option_list[user_input - 1], "---------------------")

        if user_input == option_list.index("Bootstrap Yield Curve") + 1:
            # Bootstrap YCs to derive long-term spot rates from prices
            yield_curve = BootstrapYieldCurve()
            yield_curve.add_instrument(100, 0.25, 0., 97.5)
            yield_curve.add_instrument(100, 0.5, 0., 94.9)
            yield_curve.add_instrument(100, 1.0, 0., 90.)
            yield_curve.add_instrument(100, 1.5, 8, 96., 2)
            yield_curve.add_instrument(100, 2., 12, 101.6, 2)
            y = yield_curve.get_zero_rates()
            x = yield_curve.get_maturities()
            df = pd.DataFrame({'Spot Rates': y, 'Maturity': x})
            print(df)

            # Plot YC
            #plt.plot(x, y)
            #plt.title("Zero Curve")
            #plt.ylabel("Zero Rate (%)")
            #plt.xlabel("Maturity in Years")
            #plt.show()

        elif user_input == option_list.index("Forward Rates") + 1:
            # Get forward rates from list of spot rates
            fr = ForwardRates()
            # Parameters
            spot_list = {0.25: 10.127, 0.50: 10.469, 1.00: 10.536, 1.50: 10.681, 2.00: 10.808}
            for t in spot_list.keys():
                fr.add_spot_rate(t, spot_list.values())
            fwd_list = [0] + fr.get_forward_rates() #fisrt sp
            df2 = pd.DataFrame({'Maturity': fr.spot_rates.keys(), 'Spot Rates': fr.spot_rates.values(), 'Forward Rates': fwd_list})
            print(df2)

        elif user_input == option_list.index("Bond Yield-To-Maturity") + 1:
            # Parameters
            price = float(input("Enter bond price: "))
            par = int(input("Enter par value: "))
            T = float(input("Enter time to maturity: "))
            coup = float(input("Enter coupon payment: "))
            freq = float(input("Enter frequency of coupon payment: "))
            # Get YTM of bonds
            ytm = bond_YTM(price, par, T, coup, freq)
            print("---> Bond YTM: ", ytm)

        elif user_input == option_list.index("Bond Price") + 1:
            # Parameters
            par = int(input("Enter par value: "))
            T = float(input("Enter time to maturity: "))
            ytm = float(input("Enter yield to maturity: "))
            coup = float(input("Enter coupon payment: "))
            freq = float(input("Enter frequency of coupon payment: "))
            # Get bond price
            price = bond_price(par, T, ytm, coup, freq)
            print("---> Bond Price: ", price)

        elif user_input == option_list.index("Bond Duration") + 1:
            # Parameters
            price = float(input("Enter bond price: "))
            par = int(input("Enter par value: "))
            T = float(input("Enter time to maturity: "))
            coup = float(input("Enter coupon payment: "))
            freq = float(input("Enter frequency of coupon payment: "))
            # Get bond_duration
            mod_duration = bond_moddur(price, par, T, coup, freq, dY = 0.01)
            print("---> Bond (Modified) Duration: ", mod_duration)

        elif user_input == option_list.index("Bond Convexity") + 1:
            # Parameters
            price = float(input("Enter bond price: "))
            par = int(input("Enter par value: "))
            T = float(input("Enter time to maturity: "))
            coup = float(input("Enter coupon payment: "))
            freq = float(input("Enter frequency of coupon payment: "))
            # Get bond convexity
            convexity = bond_convexity(price, par, T, coup, freq, dY = 0.01)
            print("---> Bond Convexity: ", convexity)
        else:
            print("Please choose from one of the specified options.")

    print("Thank you for using our program!")

