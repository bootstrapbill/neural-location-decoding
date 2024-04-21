# Author William Turner williamfrancisturner@gmail.com
# 

import mne         
import os 
import pickle
from decoding_funcs import decode_successive
from epoch_funcs import epoch_localizers

dataFolder = 'data/preprocessed_fifs/' 
mne.set_log_level(verbose=False) # simplify HPC output

def load_data(file):
    dataRawFile = os.path.join(dataFolder, file)
    return mne.io.read_raw_fif(dataRawFile, preload=True)

def save_data(file, scores):
    with open(file, 'wb') as f:
        pickle.dump(scores, f)
    
def run_stacked_tgm(raw, pID):
    xTrain, yTrain = epoch_localizers(raw, pID,-0.2,0.5)
    tgmScoresStacked = decode_successive(xTrain, yTrain)
    save_data('data/stacked/' + pID + '_stacked.pickle', tgmScoresStacked)

def decode(x, file):
    
    pID = file[:-4] # get ID and session number
    raw = load_data(file)
    
    run_stacked_tgm(raw, pID)
    print('stacked TGM complete  ' + pID, flush=True)

def main():
    
    files = [f for f in os.listdir(dataFolder) if not '-' in f] # get list of .fif files 
    files.sort() 
    print(files,flush=True)
    
    for x, file in enumerate(files):
        decode(x, file)
    
main()