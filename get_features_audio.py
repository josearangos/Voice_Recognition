
# coding: utf-8

# In[5]:


import IPython
import matplotlib.pyplot as plt
from matplotlib.backend_bases import RendererBase
from scipy import signal
from scipy.io import wavfile
#import soundfile as sf
import os
import numpy as np
from PIL import Image
from scipy.fftpack import fft

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


class GETFEATURES():
    def readAudios(self,path): 
        print(path)
        samplerate, data = wavfile.read(path)
        return samplerate, data
    
    def log_specgram(self,data, sample_rate, window_size=20,
                     step_size=10, eps=1e-10):
        nperseg = int(round(window_size * sample_rate / 1e3))
        noverlap = int(round(step_size * sample_rate / 1e3))
        #t Array of segment times.
        freqs,t, spec = signal.spectrogram(data,
                                        fs=sample_rate,
                                        window='hann',
                                        nperseg=nperseg,
                                        noverlap=noverlap,
                                        detrend=False)
        return freqs, np.log(spec.T.astype(np.float32) + eps)
    #Retorna array 2 filas y num_columns*num_rows en la primera columna esta la media del bloque y en la segunda la std
    
    def split_blocks(self,num_rows, num_columns, spectogram):
        features = np.zeros((2,num_columns*num_rows))
        featuresColumns = np.array_split(spectogram, num_columns,1)
        cont = 0
        for i in range(len(featuresColumns)):
            featureRow= np.array_split(featuresColumns[i], num_rows,0) 
            for j in range(len(featureRow)):
                features[0][cont]=(np.mean(featureRow[j]))
                features[1][cont]=(np.std(featureRow[j]))
                cont = cont +1
        
        return features #fila con el numero carácteristicas con la media y std
    
    def get_features_audios(self,path, num_rows, num_columns):
        sample_rate, data = self.readAudios(path)
        freqs, spectogram = self.log_specgram(data, sample_rate)
        features = self.split_blocks(num_rows, num_columns, spectogram)
        return features.flatten()        

