#!/usr/bin/env python
# coding: utf-8

# In[1]:


import librosa
import matplotlib.pyplot as plt
import librosa.display
import matplotlib
import pylab
import numpy as np


# In[10]:


def wav2melSpectIm(wav_path, targetdir,n_mels=128):
    name = wav_path.split('/')[-1].split('.')[0]
    save_path = targetdir+'/'+name
    y, sr = librosa.load(wav_path)
    whale_song, _ = librosa.effects.trim(y)
    pylab.axis('off') # no axis
    pylab.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[]) # Remove the white edge
    S = librosa.feature.melspectrogram(y=whale_song, sr=sr,n_mels=128)
    librosa.display.specshow(librosa.power_to_db(S, ref=np.max))
    pylab.savefig(save_path, bbox_inches=None, pad_inches=0)
    pylab.close()   


# In[12]:


#filename = '/home/josearangos/Documentos/Projects/Voice_Recognition/data/zero/0a2b400e_nohash_0.wav'
#path_out = '/home/josearangos/Documentos/Projects/Voice_Recognition/data/mel-spectrograms'


# In[13]:


#wav2melSpectIm(filename, path_out,n_mels=128)


# In[ ]:




