# Author William Turner williamfrancisturner@gmail.com
# 

import mne         
import os 
import pickle
from joblib import Parallel, delayed
from decoding_funcs import decode_diag_tf
from epoch_funcs import epoch_localizers, epoch_tf

dataFolder = 'data/preprocessed_fifs/' 
mne.set_log_level(verbose=False) # simplify HPC output

def load_data(file):
    dataRawFile = os.path.join(dataFolder, file)
    return mne.io.read_raw_fif(dataRawFile, preload=True)

def save_data(file, scores):
    with open(file, 'wb') as f:
        pickle.dump(scores, f)

def run_diag_tf(data, yTrain, pID): 
    xTrain = epoch_tf(data, pID)
    diagScoresTF = decode_diag_tf(xTrain, yTrain)
    save_data('data/tf/' + pID + '_tf.pickle', diagScoresTF)
    
def decode(x, file):
    pID = file[:-17] # get participant ID and session number
    raw = load_data(file)
    data, yTrain = epoch_localizers(raw, pID, -1.2, 1.5)
    del raw    
    
    run_diag_tf(data, yTrain, pID)
    print('diag TF complete  ' + pID, flush=True)

def main():
    
    files = [f for f in os.listdir(dataFolder) if f.endswith('preprocessed.fif')] # get list of .fif files 
    files.sort() 
    print(files,flush=True) # sanity check printout 
    
    Parallel(n_jobs=1)(delayed(decode)(x, file) for x, file in enumerate(files))
        
if __name__ == "__main__":
    main()
