import os
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy.fftpack import fftfreq
import librosa.core
import librosa.display

'''
def fft(rec,fmin,fmax,dx):
    result = sp.zeros(int(100000*(4/dx)))
    result[:len(rec)] = rec
    rec = result
    dt=0.006671281903963042e-9*(dx/4)
    fft_start = int(fmin*rec.shape[0]*dt)
    fft_end = int(fmax*rec.shape[0]*dt)
    fft_list = sp.fft(rec)[fft_start:fft_end]
    freq_list = fftfreq(rec.shape[0],dt)[fft_start:fft_end]
    return freq_list, abs(fft_list)
'''

if __name__=="__main__":
    data1 = sp.loadtxt("Device_001_01.txt")
    data2 = sp.loadtxt("coners-office-error-detection-35.txt")


    fig1 = plt.figure(1,[5,5])
    plots1 = [fig1.add_subplot(2,2,i) for i in range(1,5)]

    fig2 = plt.figure(2,[12,12])
    plots2 = [fig2.add_subplot(5,5,i) for i in range(1,26)]

#    fig3 = plt.figure(3,[12,12])
#    plot3 = fig3.add_subplot(111)

    D1 = np.abs(librosa.core.stft(data1,n_fft=2048))[:,15:]
    D2 = np.abs(librosa.core.stft(data2,n_fft=2048))[:,:]

    '''
    librosa.display.specshow(librosa.amplitude_to_db(D,ref=np.max)
                             , y_axis = 'log', x_axis='time')
    plt.title('Power spectrogram')
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    '''
    plots1[0].plot(range(len(data1)),data1)
    plots1[2].plot(range(len(data2)),data2)
    plots1[1].imshow(D1, aspect=0.03, vmin=0, cmap="seismic")
    plots1[3].imshow(D2, aspect=0.03, vmin=0, cmap="seismic")

    i = 0
    freqs = []
    maxs1 = []
    maxs2 = []
    for freq in sp.arange(0,1025,42):
        plots2[i].plot(range(D1.shape[1]),D1[freq,:])
        plots2[i].plot(range(D2.shape[1]),D2[freq,:])
        temp_max1 = D1[freq,:].max()
        temp_max2 = D2[freq,:].max()
        temp_freq = librosa.core.fft_frequencies(sr=48000,n_fft=2048)[freq]
        plots2[i].text(20,30000,str(temp_freq)+" Hz")
        freqs.append(temp_freq)
        maxs1.append(temp_max1)
        maxs2.append(temp_max2)
        i+=1

#    plot3.plot(freqs,maxs1,label="Device_001_05")
#    plot3.plot(freqs,maxs2, label="coners-office-error-detection-35")
#    plot3.set_xlabel("Frequency (Hz)")
#    plot3.legend()
    plt.show()
