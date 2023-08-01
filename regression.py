from scipy.optimize import curve_fit
import numpy as np

def regression(f, X, Y):
    popt, _ = curve_fit(f, X, Y)
    a = popt[0]
    ss_res = np.sum((Y - f(X, a))**2)
    ss_tot = np.sum((Y-np.mean(Y))**2)
    r_squared = 1 - (ss_res / ss_tot)
    return a, r_squared