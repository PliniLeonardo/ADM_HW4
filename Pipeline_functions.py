import pandas as pd
import numpy as np      
import matplotlib.pyplot as plt 
import scipy.io.wavfile 
import subprocess
import librosa
import librosa.display
import IPython.display as ipd
from IPython.core.display import HTML
import pickle

from pathlib import Path, PurePath   
from tqdm.notebook import tqdm

from AudioSignals import *
from minhashing import *







# Part 1-3
def dictionary_songs_json():
    '''
    create a json file with a dictionary with key=int and values= songs' names
    '''
    tracks=track_vocabulary
    songs_dictionary={}
    for i,track in enumerate(tracks):
        track=track.replace('.','-')
        songs_dictionary[i]=track.split('-')#[1]

    # create json object from dictionary
    json = json.dumps(songs_dictionary)
    # open file for writing, "w" 
    f = open("songs_dictionary.json","w")
    # write json object to file
    f.write(json)
    # close file
    f.close()

    
def shingle_dictionary_json():
    shingle_dictionary={}
    for k, v in shingles_dictionary.items():
        shingle_dictionary[str(k)]=v
        
    import json
    f = open("shingles_dictionary.json", "w")
    json.dump(shingle_dictionary, f)
    f.close()
    
# Part 4
def create_matrix(shingles_list):
    '''
    create matrix of 0 and 1 if the shingle appear in the song
    
    INPUT: shingle_list which is a list that has for every the list of the index of the shingles that appears in that song
    '''
    
    matrix=np.zeros(( len(shingles_dictionary), len(track_vocabulary)))
    for index_song in range(len(shingles_list)):
        for index_shingle in (shingles_list[index_song]):
            matrix[index_shingle,index_song]=1

    return matrix

#Part 5


N_SHINGLES = 40318

# CARDINALITY is the number of the hashfunctions we want to generate 
CARDINALITY=100
N_PRIME = 40343 #first prime number after 40318
N_PRIME = 40993 # nearest prime to 41000

def hashfunction_family(number_of_hashfunctions = CARDINALITY):
    '''
    create a family of #number_of_hashfunctions hashfunctions

    INPUT: number of hashfunctions we want to genrate
    OUTPUT: parameters  alpha and beta that univocally identificate the hashfunction family
    '''

    n=N_PRIME
    np.random.seed(seed=124)
    params_alpha = np.random.randint(0, n, number_of_hashfunctions)
    params_beta=np.random.randint(0, n, number_of_hashfunctions)

    return params_alpha,params_beta,n

# Part 6
def hash_row(index_row):
    '''
    compute the value of all the hashfunctions for the row
    INPUT= index_row
    OUTPUT=hashfunctions values
    '''
    a=params_alpha
    b=params_beta
    mod=n
    hash_values=[]
    for i in range (len(params_alpha)):
        ris=(index_row*a[i]+b[i])%mod
        hash_values.append(ris)
    return hash_values

# Part 9
def minhash_query(query):
    # load the track in memory
    track, sr = librosa.load(query, offset=OFFSET, duration=DURATION)

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
    total_number_of_triplets = len(triplet_dict)
    for idx, triplet in enumerate(track_triplets):
        if triplet not in triplet_dict.keys():
            triplet_dict[triplet] = total_number_of_triplets
            total_number_of_triplets += 1

        track_triplets[idx] = triplet_dict[triplet]

    track_triplets = np.array(track_triplets)
    
    # retrieve the hashing family
    parameters = hashfunction_family()
    
    # minhash the query
    minhashed_query = minhash_function(track_triplets, parameters)
    
    return(minhashed_query)

def minhash_function(track_triplets, parameters):
    '''
    this function computes the minhash of the input triplet list
    
    input: list of triplet id, parameters of the hashing functions
    '''
    
    number_query_words = len(track_triplets)
    
    alpha = parameters[0]
    beta = parameters[1]
    
    # the hash function is of the form
    # h(x) = (alpha*x + beta) mod N_PRIME
    
    # explicitly broadcasting the arrays to leverage numpy vectorization
    a = np.broadcast_to(alpha[np.newaxis, :], (number_query_words, CARDINALITY))
    x = np.broadcast_to(track_triplets[:, np.newaxis], (number_query_words, CARDINALITY))
    b = np.broadcast_to(beta[np.newaxis, :], (number_query_words, CARDINALITY))
    
    # calculating the hashing functions
    hash_results = np.mod(a*x + b, N_PRIME)
    
    # taking the minimum
    minhash_results = np.min(hash_results, axis = 0)
    
    return(minhash_results)

def search(query_path):
    '''
    this function finds the best matches for the input queries
    '''
    
    song_vocabulary = retrieve_track_vocabulary()
    
    #signature_matrix = retrieve_signature_matrix()
    
    best_matches = []
    for path in query_path:
        minhashed_query = minhash_query(path)
        
        similar_pairs = retrieve_similar_pairs()
        
        similarities = minhashed_jaccard_similarity(minhashed_query, signature_matrix[:, similar_pairs])
        
        best_current_match = similar_pairs[np.argmax(similarities)]
        best_current_match = song_vocabulary[best_current_match]
        
        best_matches.append(best_current_match)
    
    
    display_best_matches(best_matches)
    
    
    return

def display_best_matches(best_matches):
    
    query_labels = []
    for i in range(len(best_matches)):
        query_labels.append(f'Query {i+1}')
    
    information_df = pd.DataFrame({'Best Song Match' : best_matches})
    
    information_df.index = query_labels
    
    display(HTML(information_df.to_html(index = True)))
    
    return