import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


class Fit_Gaussian():
    """
    A class used to fit a gaussian on a histogram

    ...

    Attributes
    ----------
    data : array like
        input data. The histogram is computed over the flattened array.

    normalized : bool
        fit a normalized histogram, the default is false.

    Methods
    -------
    hist_fitting(show = False)
        Get a gaussian fitting of a histogram.
    """
    
    def __init__(self, data, normalized = False):
        """
        Parameters
        ----------
        data : array like
            input data. The histogram is computed over the flattened array.

        normalized : bool
            fit a normalized histogram, the default is false.
        """
        
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
        
        Parameters
        ----------
        bins : int, optional
        It defines the number of equal-width bins in the given range (200, by default).
        
        show : bool, optional
        Show the graph of the histogram, the default is false.
        
        Returns
        ------- 
        popt: array
        Optimal values for the parameters so that the sum of the squared residuals of the Gaussian with data is minimized
        
        X0, sigma, Y0, A of the histogram gaussian
        X0, sigma of the normalized histogram gaussian if normalized = True

        Raises
        ------
        Value Error
            if either ydata or xdata contain NaNs, or if incompatible options are used.
        Runtime Error
            if the least-squares minimization fails.
        Optimize Warning
            if covariance of the parameters can not be estimated.
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