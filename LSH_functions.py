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
import json

from pathlib import Path, PurePath   
from tqdm.notebook import tqdm

from AudioSignals import *

# Number of different triplets in our dictionary (computed after the generation of the dictionary)
N_SHINGLES = 40318

# CARDINALITY is the number of the hashfunctions we want to generate 
CARDINALITY=100

N_PRIME = 40993 # nearest prime to 41000

RANDOM_SEED = 124



###################################################################################################################################################
# Part 1-3
def dictionary_songs_json():
    '''
    create a json file with a dictionary with key=int and values= songs' names
    '''
    tracks = retrieve_track_paths()
    songs_dictionary={}
    
    for i,track in enumerate(tracks):
        track=track.replace('.','-')
        songs_dictionary[i]=track.split('-')#[1]

    with open("data/songs_dictionary.json","w") as f:
        json.dump(songs_dictionary, f)
    
    return

    
def shingle_dictionary_json():
    '''
    create a json file with a dictionary with key=triplet and values = int
    '''
    
    shingles_dictionary = load_shingles_dictionary()
    
    new_dictionary={}
    for k, v in shingles_dictionary.items():
        new_dictionary[str(k)]=v
        
    
    with open("data/shingles_dictionary.json", "w") as f:
        json.dump(new_dictionary, f)
    
    return


###################################################################################################################################################
# Part 4
def create_matrix(shingles_list):
    '''
    create matrix of 0 and 1 if the shingle appear in the song
    
    INPUT: shingle_list which is a list that has for every the list of the index of the shingles that appears in that song
    '''
    
    shingles_dictionary = load_shingles_dictionary()
    track_vocabulary = retrieve_track_vocabulary()
    shingles_list = load_shingles_list()
    
    matrix=np.zeros(( len(shingles_dictionary), len(track_vocabulary)))
    for index_song in range(len(shingles_list)):
        for index_shingle in (shingles_list[index_song]):
            matrix[index_shingle,index_song]=1

    return matrix

###################################################################################################################################################
#Part 5
def hashfunction_family(number_of_hashfunctions = CARDINALITY, n = N_PRIME, seed = RANDOM_SEED):
    '''
    create a family of #number_of_hashfunctions hashfunctions

    INPUT: number of hashfunctions we want to genrate
    OUTPUT: parameters  alpha and beta that univocally identificate the hashfunction family
    '''

    np.random.seed(seed = seed)
    params_alpha = np.random.randint(0, n, number_of_hashfunctions)
    params_beta=np.random.randint(0, n, number_of_hashfunctions)

    return params_alpha,params_beta,n

###################################################################################################################################################
# Part 6
def hash_row(index_row, a, b, n):
    '''
    compute the value of all the hash functions for the row
    INPUT= index_row, hash family parameters
    OUTPUT=hashed values
    '''
    
    hash_values=[]
    for i in range (len(a)):
        ris=(index_row*a[i]+b[i])%n
        hash_values.append(ris)
    return hash_values


def generate_signature_matrix():
    '''
    generate the signature matrix (and saves it to the disk)
    '''
    
    n_shingles = len(retrieve_track_paths())
    
    # load the shingle list
    shingles_list = load_shingles_list()
    
    # create the 0-1 matrix
    initial_matrix=create_matrix(shingles_list)
    
    # initialize the signature matrix
    signature_matrix = np.matrix(np.ones((CARDINALITY, n_shingles)) * np.inf)
    
    # generate the hash family
    a, b, n = hashfunction_family(CARDINALITY)
    
    rows_S,cols_S=signature_matrix.shape
    rows,cols=initial_matrix.shape
    
    for row_index in range(rows):
        div = np.where(initial_matrix[row_index,:]==1)
        index_non_null = div[0]

        #compute the hashfunction for this row
        hash_values=hash_row(row_index, a, b, n)

        #substitute elements in signature matrix if hash < sign[i][j]
        for i in range(len(hash_values)):
            for j in index_non_null:
                if signature_matrix[i,j]>hash_values[i]:
                    signature_matrix[i,j]=hash_values[i]

    
    # save signature matrix to the disk
    with open('data/signature_matrix.pickle', 'wb') as handle:
        pickle.dump(signature_matrix, handle)
    
    return


def retrieve_signature_matrix():
    '''
    retrieve (and returns) the signature matrix from the disk
    '''
    
    with open('data/signature_matrix.pickle', 'rb') as handle:
        signature_matrix = pickle.load(handle)
    
    return(signature_matrix)


###################################################################################################################################################
# Part 7

def minhashed_jaccard_similarity(minhashed_query, signature_matrix):
    '''
    computes the jaccard similarity between every column of the signature matrix and the query
    '''
    
    results = []
    
    for i in range(signature_matrix.shape[1]):
        results.append(jaccard(signature_matrix[:,i],minhashed_query))
    
    return(results)


def jaccard(signature_col,hashed_query):
    '''
    compute the jaccard similarity between the input columns
    '''
    
    similar=0
    for i in range(len(hashed_query)):
        if signature_col[i]==hashed_query[i]:
            similar+=1
    similarity_score=similar/len(hashed_query)
    
    return similarity_score


###################################################################################################################################################
# Part 9

def generate_hash_parameters(n, number_addings):
    '''
    randomly exctract an hash function from the universal hash family
    of number_addings parameters associated with the prime n
    '''
    
    params = np.random.randint(0, n, number_addings)
    
    return (params, n)

def retrieve_similar_pairs(query, signature_matrix, n_bands, n_rows):
    '''
    splits the signature matrix in n_bands groups of n_rows rows
    and hashes every column in the group to search for columns similar to the query
    '''
    # generate and hash function
    hash_parameters, n = generate_hash_parameters(N_PRIME, n_rows)
    
    # initialize output set of column indices
    similar_songs = set()
    
    for k in range(n_bands):
        # take a slice of the query
        band_query = np.copy(query[k*n_rows:(k+1)*n_rows])
        
        # take a slice of the signature matrix
        band = np.copy(signature_matrix[k*n_rows:(k+1)*n_rows, :])
        
        # hashes every column of the slice
        hashed = band * hash_parameters[:, np.newaxis]
        hashed = np.sum(hashed, axis = 0)
        hashed = np.mod(hashed, N_PRIME)
        
        # hashes the query
        hashed_query = np.mod(np.sum(band_query * hash_parameters, axis = 0), N_PRIME)
        
        # updates the output set
        similar_songs.update(np.where(hashed == hashed_query)[0])
    
    return list(similar_songs)


def minhash_query(query):
    '''
    this function minhashes the query
    
    input: query path
    output: minhashed query
    '''
    
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

    # load the shingles dictionary in memory
    triplet_dict = load_shingles_dictionary()
    
    # generate the triplets associated to the query
    track_triplets = generate_triplets(peaks, freq)

    # updates the dictionary of triplets with the new triplets
    total_number_of_triplets = len(triplet_dict)
    for idx, triplet in enumerate(track_triplets):
        if triplet not in triplet_dict.keys():
            triplet_dict[triplet] = total_number_of_triplets
            total_number_of_triplets += 1

        track_triplets[idx] = triplet_dict[triplet]
    
    
    # retrieve the hashing family
    parameters = hashfunction_family()
    
    # minhash the query
    track_triplets = np.array(track_triplets)
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


def search(query_path, n_bands = 5, n_rows = 20):
    '''
    this function finds the best matches for the input queries
    
    INPUT: list of query paths, number of bands for the LSH algorithm, number of rows in each band
    
    NOTE: n_bands, n_rows must be such that n_bands*n_rows = CARDINALITY
    '''
    
    # retrieve signature matrix and song vocabulary from the disk
    song_vocabulary = retrieve_track_vocabulary()
    signature_matrix = retrieve_signature_matrix()
    
    
    best_matches = []
    for path in query_path:  # computes the best match for every input query
        
        # minhashes the query
        minhashed_query = minhash_query(path)
        
        # finds the similar pairs according to the LSH algorithm
        similar_pairs = retrieve_similar_pairs(minhashed_query, signature_matrix, n_bands, n_rows)
        
        # computes the similarity between the similar pairs
        similarities = minhashed_jaccard_similarity(minhashed_query, signature_matrix[:, similar_pairs])
        
        # finds the best matche and converts it into a song title
        best_current_match = similar_pairs[np.argmax(similarities)]
        best_current_match = song_vocabulary[best_current_match]
        
        best_matches.append(best_current_match)
    
    # displays the best matches
    display_best_matches(best_matches)
    
    
    return


def display_best_matches(best_matches):
    '''
    displays the best matches
    '''
    
    query_labels = []
    for i in range(len(best_matches)):
        query_labels.append(f'Query {i+1}')
    
    information_df = pd.DataFrame({'Best Song Match' : best_matches})
    
    information_df.index = query_labels
    
    display(HTML(information_df.to_html(index = True)))
    
    return


def query_results():
    '''
    this function returns the paths of the 10 query songs, along with their real name
    '''
    
    query_paths = ['data/query/track1.wav','data/query/track2.wav','data/query/track3.wav','data/query/track4.wav','data/query/track5.wav','data/query/track6.wav','data/query/track7.wav','data/query/track8.wav','data/query/track9.wav','data/query/track10.wav']
    song_names = ['Dream_On', 'I_Want_To_Break_Free', 'October' ,'Ob-La-Di_Ob-La-Da' ,'Karma_Police' ,'Heartbreaker' ,'Go_Your_Own_Way' ,'American_Idiot' ,'Somebody', 'Black_Friday']
    
    query_labels = []
    for i in range(len(query_paths)):
        query_labels.append(f'Query {i+1}')
    
    information_df = pd.DataFrame({'Song Name' : song_names})
    
    information_df.index = query_labels
    
    return (query_paths, information_df)