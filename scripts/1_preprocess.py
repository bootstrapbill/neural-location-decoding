# Author William Turner williamfrancisturner@gmail.com
# 

import os 
import mne
from joblib import Parallel, delayed

dataFolder = 'data/raw_fifs/'
saveFolder = 'data/preprocessed_fifs/'

def load_data(file):
    
    dataRawFile = os.path.join(dataFolder, file)
    return mne.io.read_raw_fif(dataRawFile, preload=True)
    
def mark_bads(file, raw):
    
    bads = {
        "ID01_Session2_raw.fif": ['P2', 'POz'],
        "ID02_Session2_raw.fif": ['P2', 'PO4'],
        "ID03_Session2_raw.fif": ['P2', 'PO4'],
        "ID04_Session2_raw.fif": ['Pz', 'P2', 'PO4'],
        "ID05_Session2_raw.fif": ['P2', 'POz'],
        "ID06_Session2_raw.fif": ['POz', 'P2'],
        "ID07_Session1_raw.fif": ['P2', 'PO4'],
        "ID07_Session2_raw.fif": ['P2', 'PO4', 'O2'],
        "ID08_Session2_raw.fif": ['P2', 'POz'],
        "ID09_Session1_raw.fif": ['P2'],
        "ID09_Session2_raw.fif": ['P2'],
        "ID10_Session2_raw.fif": ['POz', 'PO4'],
        "ID11_Session1_raw.fif": ['P2'],
        "ID11_Session2_raw.fif": ['P2'], 
        "ID12_Session2_raw.fif": ['P2', 'PO4'],
        "ID13_Session1_raw.fif": ['P2'],
        "ID13_Session2_raw.fif": ['P2'],
        "ID15_Session1_raw.fif": ['P2', 'P4'],
        "ID15_Session2_raw.fif": ['P2', 'PO4'],
        "ID16_Session1_raw.fif": ['P2'],
        "ID16_Session2_raw.fif": ['Pz'], 
        "ID18_Session1_raw.fif": ['P2', 'POz'],
        "ID18_Session2_raw.fif": ['POz']
        }
    
    if file in bads:
        raw.info['bads'] = bads[file]
        print('marking ' + str(bads[file]) + ' in ' + str(file) + ' as bad')
        
    return raw

def pre_process(x, file, files):
    
    mne.set_log_level(verbose=False) # simplify HPC output

    raw = load_data(file)
    
    raw = mark_bads(file, raw)
    
    raw.set_eeg_reference(ref_channels=['EXG7', 'EXG8']) # re-reference to mastoids 
    raw.drop_channels(raw.info.ch_names[64:-1]) # drop externals 

    montage = mne.channels.make_standard_montage('biosemi64') 
    raw = raw.set_montage(montage)
    
    raw.interpolate_bads()
    
    # get clean ID 
    ID = file[:-8] + '.fif'
    
    # save data
    raw.save(fname = saveFolder + ID, overwrite = True)

    print('finished ' + ID, flush=True)
    
def main():
    
    files = [f for f in os.listdir(dataFolder) if f.endswith('raw.fif')] # get list of .fif files 
    files.sort() 
    print(files,flush=True) 
    
    Parallel(n_jobs=8)(delayed(pre_process)(x, file, files) for x, file in enumerate(files))

main()    