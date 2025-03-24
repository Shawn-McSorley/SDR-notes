import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal

fs = 100e3 # sample rate
fc = 25e3 # carrier frequency
A = 10 # amplitude of carrier

t_sim = 10 # total simulation time
N = int(fs*t_sim) # number of samples
t = np.arange(0, N)/fs # time vector

# generate white noise
standard_deviation = 0.01
mean = 0

random_noise = np.random.normal(mean, standard_deviation, N) # this is the additive noise we will add to the signal

# generate carrier
carrier = A*np.cos(2*np.pi*fc*t) + random_noise

# calculate and plot power spectrum
nperseg1 = 2**10
nperseg2 = 2**12
nperseg3 = 2**14
f1, Pxx1 = signal.welch(carrier, fs, nperseg=nperseg1, scaling='spectrum')
f2, Pxx2 = signal.welch(carrier, fs, nperseg=nperseg2, scaling='spectrum')
f3, Pxx3 = signal.welch(carrier, fs, nperseg=nperseg3, scaling='spectrum')

plt.figure(figsize=(10, 8))
plt.plot(f1, Pxx1, label='nperseg = 2**10')
plt.plot(f2, Pxx2, label='nperseg = 2**12')
plt.plot(f3, Pxx3, label='nperseg = 2**14')
plt.legend()
plt.yscale('log')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Power Spectrum [V**2]')
plt.xlim([fc-5e3, fc+5e3])
plt.savefig('Shawn-McSorley.github.io/assets/carrier_power_spectrum.png', dpi=200, transparent=True)

# calculate and plot power spectral density
nperseg1 = 2**10
nperseg2 = 2**12
nperseg3 = 2**14
f1, Pxx1 = signal.welch(carrier, fs, nperseg=nperseg1)
f2, Pxx2 = signal.welch(carrier, fs, nperseg=nperseg2)
f3, Pxx3 = signal.welch(carrier, fs, nperseg=nperseg3)

plt.figure(figsize=(10, 8))
plt.plot(f1, Pxx1, label='nperseg = 2**10')
plt.plot(f2, Pxx2, label='nperseg = 2**12')
plt.plot(f3, Pxx3, label='nperseg = 2**14')
plt.legend()
plt.yscale('log')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Power Spectrum [V**2/Hz]')
plt.xlim([fc-5e3, fc+5e3])
plt.savefig('Shawn-McSorley.github.io/assets/carrier_power_density.png', dpi=200, transparent=True)

# generate analytic signal
analytic_signal = signal.hilbert(carrier)
instantaneous_phase = np.unwrap(np.angle(analytic_signal)) - 2*np.pi*fc*t

down_shifted = analytic_signal * np.exp(-1j*2*np.pi*fc*t)
phase_down_shifted = np.angle(down_shifted)

nperseg = 2**15
f1, Pxx1 = signal.welch(random_noise, fs, nperseg=nperseg)
f3, Pxx3 = signal.welch(phase_down_shifted, fs, nperseg=nperseg)
scaled_additive_noise_PSD = 2 * Pxx1 / A**2

plt.figure(figsize=(10, 8))
plt.plot(f1, Pxx1, label='Generated additive noise')
plt.plot(f1, scaled_additive_noise_PSD, label='Scaled additive noise')
plt.plot(f3, Pxx3, label='Phase from down shifted')
plt.axvline(x=fc, color='k', linestyle='--', label='Carrier frequency')
plt.legend()
plt.yscale('log')
plt.xscale('log')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Power Spectrum [unit**2/Hz]')
plt.ylim([1e-11, 1e-8])
plt.savefig('Shawn-McSorley.github.io/assets/phase_noise_PSD.png', dpi=200, transparent=True)

test_modulation = 0.1 * np.sin(2*np.pi*100*t) # A slow phase modulation
A1 = 1
A2 = 0.01
carrier1 = A1*np.cos(2*np.pi*fc*t + test_modulation) + random_noise
carrier2 = A2*np.cos(2*np.pi*fc*t + test_modulation) + random_noise

analytic_signal1 = signal.hilbert(carrier1)
analytic_signal2 = signal.hilbert(carrier2)

down_shifted1 = np.angle(analytic_signal1 * np.exp(-1j*2*np.pi*fc*t))
down_shifted2 = np.angle(analytic_signal2 * np.exp(-1j*2*np.pi*fc*t))

nperseg = 2**15
f1, Pxx1 = signal.welch(down_shifted1, fs, nperseg=nperseg)
f2, Pxx2 = signal.welch(down_shifted2, fs, nperseg=nperseg)

plt.figure(figsize=(10, 8))
plt.plot(f1, Pxx1, label=f'A = {A1}')
plt.plot(f2, Pxx2, label=f'A = {A2}')
plt.legend()
plt.yscale('log')
plt.xscale('log')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Power Spectrum [rad**2/Hz]')
plt.ylim([1e-9, 2e-3])
plt.xlim([0, fc])
plt.savefig('Shawn-McSorley.github.io/assets/phase_noise_PSD_modulation.png', dpi=200, transparent=True)

test_modulation = 10 * np.sin(2*np.pi*10*t) # A slow phase modulation
A1 = 1
A2 = 0.05
carrier1 = A1*np.cos(2*np.pi*fc*t + test_modulation) + random_noise
carrier2 = A2*np.cos(2*np.pi*fc*t + test_modulation) + random_noise

analytic_signal1 = signal.hilbert(carrier1)
analytic_signal2 = signal.hilbert(carrier2)

down_shifted1 = np.unwrap(np.angle(analytic_signal1 * np.exp(-1j*2*np.pi*fc*t)))
down_shifted2 = np.unwrap(np.angle(analytic_signal2 * np.exp(-1j*2*np.pi*fc*t)))

plt.figure()
plt.plot(t, down_shifted2, label=f'A = {A2}')
plt.plot(t, down_shifted1, label=f'A = {A1}')
plt.legend(loc = 'upper right')
plt.xlabel('Time [s]')
plt.ylabel('Phase [rad]')
plt.xlim([0, 0.4])
plt.savefig('Shawn-McSorley.github.io/assets/carrier_modulation.png', dpi=200, transparent=True)
plt.show()



# Why does the phase noise PSD cutoff at the carrier frequency?
f1, Pxx1 = signal.welch(carrier, fs, nperseg=nperseg, return_onesided=False) # We will use the two-sided PSD to compare to the down-shifted analytic signal
f1 = np.fft.fftshift(f1)
Pxx1 = np.fft.fftshift(Pxx1)
f2, Pxx2 = signal.welch(analytic_signal, fs, nperseg=nperseg, return_onesided=False) # This is complex, so it will return a two sided PSD by default
f2 = np.fft.fftshift(f2)
Pxx2 = np.fft.fftshift(Pxx2)
f3, Pxx3 = signal.welch(down_shifted, fs, nperseg=nperseg, return_onesided=False) # This is complex, so it will return a two sided PSD by default
Pxx3 = np.fft.fftshift(Pxx3)
f3 = np.fft.fftshift(f3)

plt.figure()
plt.plot(f1, 4*Pxx1, label='Carrier') # The factor of 4 is because we are looking at power, and the analytic signal has a Fourier transform twice that of the real signal (in the positive frequency only)
plt.legend()
plt.yscale('log')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Power Spectrum [V**2/Hz]')

plt.figure()
plt.plot(f1, 4*Pxx1, label='Carrier')
plt.plot(f2, Pxx2, label='Analytic signal')
plt.legend()
plt.yscale('log')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Power Spectrum [V**2/Hz]')

plt.figure()
plt.plot(f1, 4*Pxx1, label='Carrier')
plt.plot(f2, Pxx2, label='Analytic signal')
plt.plot(f3, Pxx3, label='Down shifted analytic signal')
plt.legend()
plt.yscale('log')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Power Spectrum [V**2/Hz]')

plt.show()
