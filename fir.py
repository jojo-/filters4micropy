## Implementing discrete FIR ##
"""Class for second order discrete FIR filtering in micropython

Coefficients for the numerator and denominator can be determined using various filter
design software, such as Matlab"""

import math, cmath

class fir:
    """ Class for an FIR filter"""
    def __init__(self, a):
        # a - list of coefficients [a0, a1, ..., an]
        self._coeff = a

        # Initalise the buffer
        self._buffer = [0]*len(a)

        # Start the counter
        self._counter = 0

    # Function to update the filter
    def update(self, val):
        # val - The new value from the sensor
        # Store the new value
        self._buffer[self._counter] = val

        # Calculate the output
        self._y = 0
        for n in range(len(self._buffer)):
            self._y += self._buffer[n] * self._coeff[n]

        # Rotate the coefficients
        self._coeff = self.rotate(self._coeff, 1)

        # Update the counter
        self._counter = self._counter - 1 if self._counter > 0 else len(self._buffer) - 1

        # Return the output
        return self._y

    """ Function to rotate an array by k """
    def rotate(self, arr, k):
        return arr[k:]+arr[:k]

    """ Function to get the current filter value """
    def get(self):
        return self._y

# Example implementation of the filters
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import numpy as np
    import random

    # Function to give a random noise variable
    def noise(mag):
        return mag * random.gauss(1, 1)

    # Define the parameters for the fir filter
    """ These must be determined beforehand to obtain the output you want """
    a = [0.00037901217544093594,
    0.003983243842986631,
    0.010120005263499371,
    0.010266967368121263,
    -0.007027153056169479,
    -0.03675557784754312,
    -0.04509269415314178,
    0.009995897563795745,
    0.1325937532814218,
    0.26476816876515974,
    0.32220407747180513,
    0.26476816876515974,
    0.1325937532814218,
    0.009995897563795745,
    -0.04509269415314178,
    -0.03675557784754312,
    -0.007027153056169479,
    0.010266967368121263,
    0.010120005263499371,
    0.003983243842986631,
    0.00037901217544093594]

    # Initialise the filter
    filter = fir(a)

    # Create the dummy dataset
    N = 1024 # Number of samples
    Fs = 500 # Sample rate (samples/sec)
    Ts = 1 / Fs # Sample period (sec)

    # Time variable
    t = np.linspace(0.0, N*Ts, N)

    # Example output - two sinusoids
    x = list(np.sin(5.0 * 2.0*np.pi*t) + 0.5*np.sin(2.0 * 2.0*np.pi*t))

    # Add some Gaussian noise
    y = [output + noise(0.1) for output in x]

    # Start an empty list
    filtered = []

    # Cycle through the output and filter
    for y_val in y:
        # Update the filter
        filter.update(y_val)

        # Get and store the new filtered value
        filtered_val = filter.get()
        filtered.append(filtered_val)

    # Plot the results
    plt.figure(1)
    plt.plot(t, y) # Noisy signal
    plt.plot(t, filtered) # Filtered signal
    plt.show()
