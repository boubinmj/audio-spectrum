import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write
from scipy.io.wavfile import read
import matplotlib.pyplot as plt
from scipy.fft import fft
import csv

filename = 'Plane_Stopped.csv'

fs = 44100
seconds = 3
sounds_fft = []
freqs_max = 1000

for i in range(0,10):

    myrecording = sd.rec(int(seconds*fs),samplerate=fs, channels=2)
    sd.wait()
    write('output.wav', fs, myrecording)

    freq,recording = read('output.wav')
    plt.plot(range(1,seconds*fs), recording[1:,0])
    #plt.show()

    rec_fft = fft(recording[1:,0])

    L = len(recording[1:,0])

    power = np.abs(rec_fft/L)**2
    power1 = power[0:(int(L/2)+1)]
    power1[2:-1] = 2*power1[2:-1]

    freqs = fs*np.arange(0,L/2,1)/L
    
    if(i==0):
        sounds_fft.append(freqs[1:freqs_max])

    plt.plot(power1[1:3000])
    #plt.show()

    sounds_fft.append(power1[1:freqs_max])
    
import pandas as pd

df = pd.DataFrame(sounds_fft)
df.to_csv(filename, header=False, index=False)
