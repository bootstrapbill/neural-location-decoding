# Author William Turner williamfrancisturner@gmail.com
# 

import mne         
import os 
import pickle
from joblib import Parallel, delayed
from decoding_funcs import (decode_diag, error_by_pos, decode_TGM, searchlight)
from epoch_funcs import epoch_localizers, get_just_eoochs

dataFolder = 'data/preprocessed_fifs/' 
mne.set_log_level(verbose=False) # simplify HPC output

def load_data(file):
    dataRawFile = os.path.join(dataFolder, file)
    return mne.io.read_raw_fif(dataRawFile, preload=True)

def save_data(file, scores):
    with open(file, 'wb') as f:
        pickle.dump(scores, f)

def run_diag(xTrain, yTrain, pID): 
    diagScores = decode_diag(xTrain, yTrain)
    save_data('data/diag/' + pID + '_diag.pickle', diagScores)
          
def run_positions(xTrain, yTrain, pID):
    posScores = error_by_pos(xTrain, yTrain)
    save_data('data/pos/' + pID + '_pos.pickle', posScores)
                
def run_tgm(xTrain, yTrain, pID):
    tgmScores = decode_TGM(xTrain, yTrain)
    save_data('data/tgm/' + pID + '_tgm.pickle', tgmScores)

def run_search(stimEpochs, pID):                
    searchScores = searchlight(stimEpochs)
    save_data('data/search/' + pID + '_search.pickle', searchScores)

def decode(x, file):
    pID = file[:-17] # get participant ID and session number
    raw = load_data(file)
    
    xTrain, yTrain = epoch_localizers(raw, pID)
    del raw # reduce memory overhead 
    
    run_diag(xTrain, yTrain, pID)
    print('diag complete  ' + pID, flush=True)

    run_positions(xTrain, yTrain, pID)
    print('pos decoding complete  ' + pID, flush=True)

    run_tgm(xTrain, yTrain, pID)
    print('tgm complete  ' + pID, flush=True)

    del xTrain, yTrain # reduce memory overhead 
    
    raw = load_data(file)
    stimEpochs = get_just_eoochs(raw, pID)
    stimEpochs = stimEpochs[0] # CHECK IF NECESSARY! 
    del raw # try to reduce memory overhead (NOTE: This may not work since stimEpochs contains ref to raw...)
    
    run_search(stimEpochs, pID)
    print('searchlight complete  ' + pID, flush=True)

def main():
    
    files = [f for f in os.listdir(dataFolder) if f.endswith('preprocessed.fif')] # get list of .fif files 
    files.sort() 
    print(files,flush=True) # sanity check printout 
    
    Parallel(n_jobs=1,backend='multiprocessing')(delayed(decode)(x, file) for x, file in enumerate(files))

if __name__ == "__main__":
    main()
