class AverageFilter:

    def __init__(self):
        self.__k = 1
        self.__val = 0

    def update(self, value):
        alpha = (self.__k - 1) / self.__k
        self.__val = alpha * self.__val + (1 - alpha) * value
        self.__k = self.__k + 1
        return self.__val

    def get(self):
        return self.__val

class MovingAverageFilter:

    def __init__(self, window = 1):
        self.__val = 0
        self.__window = window
        self.__data = None

    def update(self, value):

        if self.__data is None:
            self.__data = [value] * self.__window
            self.__val = value

        self.__data.pop(0)
        self.__data.append(value)
        self.__val = self.__val + (value - self.__data[0]) / self.__window

        return self.__val

    def get(self):
        return self.__val

class LowPass1Filter:

    def __init__(self, alpha):
        self.__val = None
        self.__alpha = alpha

    def update(self, value):

        if self.__val == None:
            self.__val = value

        self.__val = self.__alpha  * self.__val + (1 - self.__alpha) * value
        
        return self.__val

    def get(self):
        return self.__val
    
if __name__ == "__main__":

    import numpy as np
    import matplotlib.pyplot as plt
    import random
    
    ### TEST AVERAGE
    
    def get_volt():
        return 14.0 + 4.0 * random.gauss(1,1)
   
    t = range(0,100,1)
    n = len(t)
    mu = AverageFilter()
    
    x_l = list()
    mu_l = list()
    
    for k in range(n):

        x = get_volt()
        mu.update(x)
        
        x_l.append(x)
        mu_l.append(mu.get())

    plt.figure(0)
    plt.plot(t,x_l)
    plt.plot(t,mu_l)
    plt.show()

    ### TEST MOVING AVERAGE

    def get_volt2(k):
        return 14.0 + 4.0 * random.gauss(1,1) + k * random.gauss(1,1)

    mu2 = MovingAverageFilter(10)

    x2_l = list()
    mu2_l = list()

    for k in range(n):

        x = get_volt2(k)
        mu2.update(x)

        x2_l.append(x)
        mu2_l.append(mu2.get())

    print(x2_l)
    print(mu2_l)
        
    plt.figure(1)
    plt.plot(t,x2_l)
    plt.plot(t,mu2_l)
    plt.show()

    ### TEST FIRST ORDER LOW PASS FILTER
    
    mu3 = LowPass1Filter(0.7)

    x3_l = list()
    mu3_l = list()
    
    for k in range(n):

        x = get_volt2(k)
        mu3.update(x)

        x3_l.append(x)
        mu3_l.append(mu3.get())

    print(x3_l)
    print(mu3_l)
        
    plt.figure(2)
    plt.plot(t,x3_l)
    plt.plot(t,mu3_l)
    plt.show()

        
