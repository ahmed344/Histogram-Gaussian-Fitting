import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


class Fit_Gaussian():
    
    def __init__(self, data, normalized = False):
        
        self.data = data                     # The image to the Gaussian on.
        self.normalized = normalized       # Using normalized Gaussian. 

    # Define a gaussian
    def Gauss(x, x0, sigma, y0, A):
        return y0 + A * np.exp(-(x - x0)**2 / (2 * sigma**2))
    
    # Define a normalized gaussian
    def Gauss_normalized(x, x0, sigma):
        return (1/np.sqrt(2 * np.pi * sigma**2)) * np.exp(-(x - x0)**2 / (2 * sigma**2))

    # Fit a gaussian on a histogram
    def hist_fitting(self, bins = 200, show = False):
        """Get a gaussian fitting of a histogram.
        Parameter:
            data - as numpy array
        Returns: 
            X0, sigma, Y0, A - of the histogram gaussian
            or
            X0, sigma - of the normalized histogram gaussian
        """
        # Make a histogram
        if self.normalized == True:
            n, bins_ = np.histogram(self.data, bins=bins, density = True)
        else:
            n, bins_ = np.histogram(self.data, bins=bins)


        # Data
        x = np.linspace(bins_.min(),bins_.max(),bins_.shape[0]-1)
        y = n

        # Apply the fitting 
        if self.normalized == True:
            popt,pcov = curve_fit(Fit_Gaussian.Gauss_normalized, x, y,
                                  p0 = (x.max()/2, x.max()/3),
                                  bounds = (0, [x.max(), x.max()/2]))
        else:
            popt,pcov = curve_fit(Fit_Gaussian.Gauss, x, y,
                                  p0 = (x.max()/2, x.max()/3, 0, 1/np.sqrt(2 * np.pi * (x.max()/3)**2)),
                                  bounds = (0, [x.max(), x.max()/2, np.inf, np.inf]))            

        # Display the results
        if show == True:
            plt.figure(figsize=(8,5))
            
            if self.normalized == True:
                plt.plot(x, Fit_Gaussian.Gauss_normalized(x, *popt), 'r-',
                         label='Gauss: $x_0$ = {:.4f}, $\sigma$ = {:.4f}'.format(*popt))
            else:
                plt.plot(x, Fit_Gaussian.Gauss(x, *popt), 'r-',
                         label='Gauss: $x_0$ = {:.4f}, $\sigma$ = {:.4f}, $Y_0$ = {:.4f}, A = {:.4f}'.format(*popt))
            
            plt.plot(x, y, 'b+:', label='data')            
            plt.legend()
            plt.title('Histogram Gaussian')
            plt.xlabel('value')
            plt.ylabel('frequency')
            plt.grid()

        return popt  #(X0, sigma, Y0, A) or just (X0, sigma) for normalized Gaussian