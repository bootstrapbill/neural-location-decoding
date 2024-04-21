# Author William Turner williamfrancisturner@gmail.com
# 

import os 
import mne
import pickle
from epoch_funcs import epoch_loc_and_motion, create_synthetic
from decoding_funcs import LDA_map, LDA_map_synth

dataFolder = 'data/preprocessed_fifs/' 
mne.set_log_level(verbose=False) # simplify HPC output

def load_data(file):
    dataRawFile = os.path.join(dataFolder, file)
    return mne.io.read_raw_fif(dataRawFile, preload=True)

def save_data(file, scores):
    with open(file, 'wb') as f:
        pickle.dump(scores, f)
        
def run_decoding(x, file):
    
    pID = file[:-4]
    raw = load_data(file)

    xTrain, yTrain, xTest, yTest = epoch_loc_and_motion(raw, pID)   
    scores = LDA_map(xTrain, yTrain, xTest, yTest)
    save_data('data/LDA/' + pID + '_LDA.pickle', scores)
    
    xTrain, yTrain = epoch_loc_and_motion(raw, pID, -0.2, 1.1, synth = True)
    xTrain, xTest, yTrain, yTest = create_synthetic(xTrain, yTrain)
    scores = LDA_map_synth(xTrain, yTrain, xTest, yTest, synth=True)
    save_data('data/LDA/' + pID + '_synth.pickle', scores)
    
def main():
    
    files = [f for f in os.listdir(dataFolder) if not '-' in f] # get list of .fif files 
    files.sort() 
    print(files,flush=True)
    
    for x, file in enumerate(files):
        run_decoding(x, file)
                
main()