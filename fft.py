# Implementing discrete FFT for micropython.
#
# Some doc:
# https://stackoverflow.com/questions/25735153/plotting-a-fast-fourier-transform-in-python
# https://www.reddit.com/r/Python/comments/804w3l/need_help_understanding_numpy_fft/
# https://github.com/numpy/numpy/blob/v1.14.0/numpy/fft/helper.py#L130-L175
# https://jakevdp.github.io/blog/2013/08/28/understanding-the-fft/
# https://github.com/peterhinch/micropython-fft/blob/master/algorithms.py

import math, cmath

# Naive, recursive version of the FFT
# see https://forum.micropython.org/viewtopic.php?f=2&t=208&hilit=fft
def fft_slow(x):
    N = len(x)
    if N <= 1:
        return x
    even = fft_slow(x[0::2])
    odd =  fft_slow(x[1::2])
    T = [cmath.exp(-2j*cmath.pi*k/N)*odd[k] for k in range(N//2)]
    return [even[k] + T[k] for k in range(N//2)] + [even[k] - T[k] for k in range(N//2)]


# Faster implementation
# see https://github.com/peterhinch/micropython-fft/blob/master/algorithms.py
def fft(x, forward=True, scale=False, real=True):
    
    res = list(x)    
    n = len(res)
    
    # Calculate the number of points: n = 2**m    
    log2_n = math.log2(n)
    assert log2_n % 1 == 0, "Lenght of sample not a power of 2!"
    m = int(log2_n)
    
    # Do the bit reversal
    i2 = n >> 1  # i2 = n / 2
    j = 0   
    for i in range(n-1):
        if i<j: 
            res[i], res[j] = res[j], res[i]
        k = i2
        while (k <= j):
            j -= k
            k >>= 1
        j+=k
        
    # Compute the FFT
    c = 0j-1
    l2 = 1
    for l in range(m): 
        l1 = l2
        l2 <<= 1
        u = 0j+1
        for j in range(l1):
            for i in range(j, n, l2):
                i1 = i+l1
                t1 = u * res[i1]
                res[i1] = res[i] - t1
                res[i] += t1
            u *= c
        ci = math.sqrt((1.0 - c.real) / 2.0) # Generate complex roots of unity
   
        if forward: 
            ci = -ci                         # for forward transform
            
        cr = math.sqrt((1.0 + c.real) / 2.0) # phi = -pi/2 -pi/4 -pi/8...
        c = cr + ci*1j                       #complex(cr,ci)
    
    # Scaling for forward transform
    if (scale and forward):
        for i in range(n):
            res[i] = res[i] / math.sqrt(n)
    
    if real:
        return res[:n//2 + 1]
    return res


# Computes the discrete FFT sample frequencies
# see https://github.com/numpy/numpy/blob/v1.14.0/numpy/fft/helper.py#L130-L175
def fft_freq_real(w, d=1.0):
    # w : window length
    # d : sample spacing
    
    val = 1.0 / (w * d)
    N   = w // 2 + 1
    res = range(0,N)
    return [r * val for r in res]
 
    
#### Tests (just making sure everything work as expected
if __name__ == '__main__':
    
    import matplotlib.pyplot as plt
    import numpy as np
    import timeit
    
    # number of samples
    N = 1024
    # sample spacing
    T = 1.0 / 800.0
    # creating the dummy dataset
    x = np.linspace(0.0, N*T, N)
    y = list(np.sin(50.0 * 2.0*np.pi*x) + 0.5*np.sin(80.0 * 2.0*np.pi*x))
    
    # fft and computing the frequencies
    out = fft(y)
    out_f = fft_freq_real(N, T)
    
    # some plots
    _, ax = plt.subplots(2)
    ax[0].plot(x, y)
    ax[1].plot(out_f, [abs(o) for o in out])
    plt.show()
    
    # Testing the result against numpy's implementation and a slower implementation for complex inputs (general case)
    x = list(np.random.random(N))
    print("Test 1: " + str(np.allclose(fft(x, real=False), np.fft.fft(x))))
    print("Test 2: " + str(np.allclose(np.fft.fft(x), fft_slow(x))))
    print("Test 3: " + str(np.allclose(fft(x, real=False), fft_slow(x))))
    
    # Testing real inputs
    print("Test 4: " + str( np.allclose( fft(x), np.fft.rfft(x))))
    
    # Testing frequencies real
    print("Test 5: " + str(np.allclose(fft_freq_real(60, 1/60), np.fft.rfftfreq(60,1/60))))
    
    # Timing
    print("Timing fft (real): " + str(timeit.timeit('fft(x)', globals=globals(), number=100)))
    print("Timing fft_slow: " + str(timeit.timeit('fft_slow(x)', globals=globals(), number=100)))
    print("Timing np.fft.rfft: " + str(timeit.timeit('np.fft.rfft(x)', globals=globals(), number=100)))
    
    # Another application: finding the base frequency inside noise
    # see http://nickdnickd.com/blog/2018/01/16/Let-There-Be-Sound  
    two_pi = 2 * np.pi
    phase = np.pi/2
    frequency = 440    # "A" note in Hz
    duration_s = 0.03125 / 2
    sample_rate = 2**15  # Hz or samples/second
    
    def createCosineAudioSignal(freq, duration_s, sample_rate):
        two_pi = 2 * np.pi
        x_samples = range(int(duration_s*sample_rate))
        x_time = [(float(x_sample) / float(sample_rate)) for x_sample in x_samples]
        cosine_signal = np.cos([two_pi * frequency * time_el + phase for time_el in x_time])
        return (cosine_signal, x_samples, x_time)
    
    def addAwgNoiseToSignal(input_discr_signal, snr_dB):
        """Calculates signal power and generates additive white gaussian noise to match SNR param"""
        sigp_dB = 10 * (np.log10(np.linalg.norm(input_discr_signal,2))**2) /\
                             len(input_discr_signal)
        noisep_dB = sigp_dB - snr_dB
        noisep = 10**(noisep_dB/10)
        noise = np.sqrt(noisep)*np.random.randn(len(input_discr_signal))
        return [sig + noise for (sig,noise) in zip(input_discr_signal, noise)]
      
    # Base signal
    x_samples = range(int(duration_s*sample_rate))
    x_time = [(float(x_sample) / float(sample_rate)) for x_sample in x_samples]
    (y_0, x_samples, x_time) = createCosineAudioSignal(frequency, duration_s, sample_rate)
    
    # Construct a new signal based off y_0 but with added noise to match the snr
    y_1 = addAwgNoiseToSignal(y_0, -15)
    
    # FFT
    F_y_1 = fft(y_1)
    freq  = fft_freq_real(len(x_samples), 1 / sample_rate)
    
    # Plots
    _, ax2 = plt.subplots(3)
    ax2[0].plot(x_time, y_0)
    ax2[1].plot(x_time, y_1)
    ax2[2].plot(freq, [abs(y) for y in F_y_1] )
    ax2[2].set_xlim(xmax=2*frequency, xmin = 0)
    plt.show()