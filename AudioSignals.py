import numpy as np      
import matplotlib.pyplot as plt 
import scipy.io.wavfile 
import subprocess
import librosa
import librosa.display
import IPython.display as ipd

from pathlib import Path, PurePath   
from tqdm.notebook import tqdm

import pickle

'''
Library of utility functions from AudioSignals.ipynb, plus some add-ons
'''

N_TRACKS = 1413
HOP_SIZE = 512
OFFSET = 1.0
DURATION = 30

###################################################################################################################################################

# Functions from AudioSignals.ipynb

def convert_mp3_to_wav(audio:str) -> str:  
    """Convert an input MP3 audio track into a WAV file.

    Args:
        audio (str): An input audio track.

    Returns:
        [str]: WAV filename.
    """
    if audio[-3:] == "mp3":
        wav_audio = audio[:-3] + "wav"
        if not Path(wav_audio).exists():
                subprocess.check_output(f"ffmpeg -i {audio} {wav_audio}", shell=True)
        return wav_audio
    
    return audio

def plot_spectrogram_and_peaks(track:np.ndarray, sr:int, peaks:np.ndarray, onset_env:np.ndarray) -> None:
    """Plots the spectrogram and peaks 

    Args:
        track (np.ndarray): A track.
        sr (int): Aampling rate.
        peaks (np.ndarray): Indices of peaks in the track.
        onset_env (np.ndarray): Vector containing the onset strength envelope.
    """
    times = librosa.frames_to_time(np.arange(len(onset_env)),
                            sr=sr, hop_length=HOP_SIZE)

    plt.figure()
    ax = plt.subplot(2, 1, 2)
    D = librosa.stft(track)
    librosa.display.specshow(librosa.amplitude_to_db(np.abs(D), ref=np.max),
                            y_axis='log', x_axis='time')
    plt.subplot(2, 1, 1, sharex=ax)
    plt.plot(times, onset_env, alpha=0.8, label='Onset strength')
    plt.vlines(times[peaks], 0,
            onset_env.max(), color='r', alpha=0.8,
            label='Selected peaks')
    plt.legend(frameon=True, framealpha=0.8)
    plt.axis('tight')
    plt.tight_layout()
    plt.show()

def load_audio_peaks(audio, offset, duration, hop_size):
    """Load the tracks and peaks of an audio.

    Args:
        audio (string, int, pathlib.Path or file-like object): [description]
        offset (float): start reading after this time (in seconds)
        duration (float): only load up to this much audio (in seconds)
        hop_size (int): the hop_length

    Returns:
        tuple: Returns the audio time series (track) and sampling rate (sr), a vector containing the onset strength envelope
        (onset_env), and the indices of peaks in track (peaks).
    """
    try:
        track, sr = librosa.load(audio, offset=offset, duration=duration)
        onset_env = librosa.onset.onset_strength(track, sr=sr, hop_length=hop_size)
        peaks = librosa.util.peak_pick(onset_env, 10, 10, 10, 10, 0.5, 0.5)
    except Error as e:
        print('An error occurred processing ', str(audio))
        print(e)

    return track, sr, onset_env, peaks


def track_conversion():
    '''
    Converts every mp3 track to wav
    '''
    data_folder = Path("data/mp3s-32k/")
    mp3_tracks = data_folder.glob("*/*/*.mp3")
    
    for track in tqdm(mp3_tracks, total=N_TRACKS):
        convert_mp3_to_wav(str(track))
    
    return

###################################################################################################################################################


# Add-ons

def retrieve_track_paths():
    '''
    returns a list that contains all the track paths
    '''
    
    # we read the track paths from the all.list file in the dataset
    with open("data/wav_list.txt", "r") as file:
        all_paths = file.readlines()
    
    # we add the parent path to get to the current directory
    for i, path in enumerate(all_paths):
        all_paths[i] = path.strip()
    
    return(all_paths)


def retrieve_track_vocabulary():
    '''
    returns a list that contains all the track names
    '''
    
    # we get the names from the paths
    paths = retrieve_track_paths()
    
    names = []
    for path in paths:
        filename = path.split('/')[-1]
        song_name = filename[3:-4]
        names.append(song_name)
    
    return(names)


def create_wav_list():
    '''
    create the list of all wav files path and saves it into a file
    '''
    
    data_folder = Path("data/mp3s-32k/")
    tracks = data_folder.glob("*/*/*.wav")
    tracks = list(map(str, tracks))
    
    with open("data/wav_list.txt", "w") as wav_file:
        for path in tracks:
            wav_file.write(path + '\n')
    
    return


def indexed_plot_spectrogram_and_peaks(idx, offset = OFFSET, duration = DURATION):
    '''
    plots the spectrogram for the i^th track (according to the all.list ordering)
    NOTE: idx goes from 0 to N_TRACKS-1
    '''
    
    paths = retrieve_track_paths()
    
    audio = paths[idx]
    track, sr, onset_env, peaks = load_audio_peaks(audio, offset, duration, HOP_SIZE)
    plot_spectrogram_and_peaks(track, sr, peaks, onset_env)
    
    return


def bins_to_time(bins, sr = 22050, hop_length = HOP_SIZE):
    '''
    convert a series of bin numbers to time (in seconds)
    '''
    
    times = librosa.frames_to_time(bins, sr = sr, hop_length = hop_length)
    
    return(times)
    
    
def bins_to_freq(bins = None, sr = 22050, n_fft = 2048):
    '''
    convert a series of bin numbers to frequencies
    '''
    
    freq = librosa.fft_frequencies(sr = sr, n_fft = n_fft)
    
    return(freq)


def generate_peak_triplets():
    '''
    here we generate the peak triplets for each track and the associated dictionary
    and we save them in two files
    '''
    
    track_paths = retrieve_track_paths()
    
    triplet_dict = {}
    total_number_of_triplets = 0
    
    triplets_list = []
    
    for idx, audio in tqdm(enumerate(track_paths), total = N_TRACKS):
        
        # load the track in memory
        track, sr = librosa.load(audio, offset=OFFSET, duration=DURATION)
        
        # analyze the structure of the song
        onset_env = librosa.onset.onset_strength(track, sr=sr, hop_length = HOP_SIZE)
        
        
        # retrieve the peak indices (these corresponds to the time bins of the peaks)
        peaks = librosa.util.peak_pick(onset_env, 10, 10, 10, 10, 0.5, 0.5)
        
        
        # generate the STFT matrix
        D = librosa.stft(track)
        
        # select only the peaks
        D = D[:,peaks]
        
        # select the intensity of every element
        D = np.abs(D)
        
        # retrieve the frequency associated to the maximum
        freq = np.argmax(D, axis = 0)
        
        
        # generate the triplets
        track_triplets = generate_triplets(peaks, freq)
        
        # create the dictionary of triplets
        for idx, triplet in enumerate(track_triplets):
            if triplet not in triplet_dict.keys():
                triplet_dict[triplet] = total_number_of_triplets
                total_number_of_triplets += 1
            
            track_triplets[idx] = triplet_dict[triplet]
        
        
        triplets_list.append(track_triplets)
    
    
    
    # Store shingles list
    with open('data/shingles_list.pickle', 'wb') as handle:
        pickle.dump(triplets_list, handle)
    
    # Store shingles dictionary
    with open('data/shingles_dictionary.pickle', 'wb') as handle:
        pickle.dump(triplet_dict, handle)
    
    
    return


def generate_triplets(peaks, freq):
    '''
    generate the track triplets starting from the time and frequency bin indices
    '''
    
    track_triplets = []
    for i in range(len(freq)-1):
        current_freq = freq[i]
        next_freq = freq[i+1]
        time_diff = peaks[i+1] - peaks[i]
        
        track_triplets.append((current_freq, time_diff, next_freq))
    
    return(track_triplets)
    
    
def load_shingles_list():
    '''
    loads the shingles list in memory
    '''
    # Load data (deserialize)
    with open('data/shingles_list.pickle', 'rb') as handle:
        shingles_list = pickle.load(handle)
    
    return(shingles_list)


def load_shingles_dictionary():
    '''
    loads the shingles list in memory
    '''
    # Load data (deserialize)
    with open('data/shingles_dictionary.pickle', 'rb') as handle:
        shingles_dictionary = pickle.load(handle)
    
    return(shingles_dictionary)