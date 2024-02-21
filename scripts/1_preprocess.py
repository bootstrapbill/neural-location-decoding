# Author William Turner williamfrancisturner@gmail.com
# 

import os 
import mne
from joblib import Parallel, delayed

mne.set_log_level(verbose=False) # set up logging (to simplify HPC output)
dataFolder = 'data/raw_fifs/'
saveFolder = 'data/preprocessed_fifs/'

def load_data(file):
    
    dataRawFile = os.path.join(dataFolder, file)
    return mne.io.read_raw_fif(dataRawFile, preload=True)
    
def mark_bads(file, raw):
    
    bads = {
        "A_ID23_GC_Session1_1.bdf": ['P2'],
        "A_ID23_GC_Session2_1.bdf": ['P2'],
        "A_ID25_BP_Session1_1.bdf": ['P2'],
        "A_ID26_AC_Session2_1.bdf": ['P2', 'PO4'],
        "A_ID29_BM2_Session1_1.bdf": ['P2', 'P4'],
        "A_ID30_YS_Session1_1.bdf": ['P2'],
        "A_ID30_YS_Session2_1.bdf": ['Pz'],
        "ID10_LC_Session2.bdf": ['P2', 'POz'],
        "ID11_SY_Session2.bdf": ['P2', 'PO4'],
        "ID12_AQ_Session2.bdf": ['P2', 'PO4'],
        "ID13_KN_Session2.bdf": ['Pz', 'P2', 'PO4'],
        "ID15_ET_Session2.bdf": ['P2', 'POz'],
        "ID17_CS_Session2.bdf": ['POz', 'P2'],
        "ID18_TL_Session1.bdf": ['P2', 'PO4'],
        "ID18_TL_Session2.bdf": ['P2', 'PO4', 'O2'],
        "ID22_MR_Session2.bdf": ['P2', 'POz'],
        "ID24_AS_Session1.bdf": ['POz', 'PO4'],
        "ID25_BP_Session2.bdf": ['P2'], 
        "ID27_BM_Session1.bdf": ['P2'],
        "ID27_BM_Session2.bdf": ['P2'],
        "ID29_BM2_Session2.bdf": ['P2', 'PO4'],
        "ID32_MJ_Session1.bdf": ['P2', 'POz'],
        "ID32_MJ_Session2.bdf": ['POz']
        }
    
    if file in bads:
        raw.info['bads'] = bads[file]
    
    return raw

def pre_process(x, file, files):

    raw = load_data(file)
    
    raw = mark_bads(file, raw)
    
    raw.set_eeg_reference(ref_channels=['EXG7', 'EXG8']) # re-reference to mastoids 
    raw.drop_channels(raw.info.ch_names[64:-1]) # drop externals 

    montage = mne.channels.make_standard_montage('biosemi64') 
    raw = raw.set_montage(montage)
    
    raw.interpolate_bads()
    
    # get clean ID 
    ID = file[:-8] + '_preprocessed.fif'
    
    # save data
    raw.save(fname = saveFolder + ID, overwrite = True)

    print('finished ' + ID, flush=True)
    
def main():
    
    files = [f for f in os.listdir(dataFolder) if f.endswith('raw.fif')] # get list of .fif files 
    files.sort() 
    print(files,flush=True) # sanity check printout 
    
    Parallel(n_jobs=8)(delayed(pre_process)(x, file, files) for x, file in enumerate(files))

if __name__ == "__main__":
    main()
    