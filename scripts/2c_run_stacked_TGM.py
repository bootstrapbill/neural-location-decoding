# Author William Turner williamfrancisturner@gmail.com
# 
# Run a stacked TGM analysis, inspired by 
# King & Wyart, and Gwilliams et al. 

import mne         
import os 
import pickle
from joblib import Parallel, delayed
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
    x, y = epoch_localizers(raw, pID,-0.2,0.5)
    tgmScoresStacked = decode_successive(x, y)
    save_data('data/stacked/' + pID + '_stacked.pickle', tgmScoresStacked)

def decode(x, file):
    pID = file[:-17] # get participant ID and session number
    raw = load_data(file)
    run_stacked_tgm(raw, pID)
    print('stacked TGM complete  ' + pID, flush=True)

def main():
    
    files = [f for f in os.listdir(dataFolder) if f.endswith('preprocessed.fif')] # get list of .fif files 
    files.sort() 
    print(files,flush=True) # sanity check printout 
    
    Parallel(n_jobs=1,backend='multiprocessing')(delayed(decode)(x, file) for x, file in enumerate(files))

if __name__ == "__main__":
    main()
