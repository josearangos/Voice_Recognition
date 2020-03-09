import librosa
import matplotlib.pyplot as plt
import librosa.display
import matplotlib
import pylab
import numpy as np
from sklearn import preprocessing
import sklearn as sk



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



def wavToMelSpect(wav_path,n_mels=120):
    name = wav_path.split('/')[-1].split('.')[0]
    y, sr = librosa.load(wav_path, duration=1.0)
    whale_song, _ = librosa.effects.trim(y) 
    spect = pad_audio(whale_song)      
    spect = librosa.feature.melspectrogram(y=spect, sr=sr,n_mels=128) 
    spect_norm=sk.preprocessing.minmax_scale(spect, axis=0) 
    return spect_norm
 

def pad_audio(samples):
    L = 22050
    if len(samples) >= L: return samples
    else: return np.pad(samples, pad_width=(L - len(samples), 0), mode='constant', constant_values=(0, 0))   
    