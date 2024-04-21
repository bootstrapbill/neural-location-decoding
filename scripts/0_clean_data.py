# Author William Turner williamfrancisturner@gmail.com
# 

import os 
import mne

mne.set_log_level(verbose=False) # set up logging (to simplify HPC output)

dataFolder = 'data/raw_bdfs/'
saveFolder = 'data/raw_fifs/'

# The first 9 participants (not included below) were pilot. 
# 23 more were tested, but 5 did not complete 2 sessions. 
# Final sample is 18 participants w/ 2 sessions.
clean_id = {
            "ID10": "01", "ID11": "02", "ID12": "03",
            "ID13": "04", "ID15": "05", "ID17": "06",
            "ID18": "07", "ID22": "08", "ID23": "09",
            "ID24": "10", "ID25": "11", "ID26": "12",
            "ID27": "13", "ID28": "14", "ID29": "15",
            "ID30": "16", "ID31": "17", "ID32": "18",
            }

def clean_data(x, file, files):
    
    """Load raw .bdfs and concatenate when multiple files 
    were created in a single session"""
    
    if x > 0:
        if file[:18] == files[x-1][:18]:
            print('skipping' + str(file))
            return
        
    dataRawFile = os.path.join(dataFolder, file)
    raw = mne.io.read_raw_bdf(dataRawFile, preload=True)
    
    # concatenate data from split files
    for lookForward in [1,2]:
        if x + lookForward < len(files): 
            if files[x + lookForward][:18] == file[:18]:
        
                dataRawFile = os.path.join(dataFolder, files[x + lookForward])
                raw2 = mne.io.read_raw_bdf(dataRawFile, preload=True)
                
                print('concatenating ' + str(files[x + lookForward]) + ' to ' + str(file), flush = True) 
                raw = mne.io.concatenate_raws([raw, raw2]) 

    if file == 'ID10_LC_Session1.bdf': # remove data from testing triggers prior to actual exp
        raw.crop(tmin = 1000)
 
    if file[0] == 'A':  # extract file + session labels 
        ID = file[2:6]
        Session = file[-7]
    else: 
        ID = file[0:4]
        Session = file[-5]
    
    ID = str(clean_id[ID])  # get clean ID 

    raw.save(fname = saveFolder + 'ID' + ID + '_Session' + Session + '_raw.fif', overwrite = True)

    print('finished ' + ID + ' Session' + Session, flush=True)

def main(): 
    
    files = [f for f in os.listdir(dataFolder) if f.endswith('.bdf')] # get list of .bdf files 
    files.sort() 
    print(files,flush=True)

    for x, file in enumerate(files):
        clean_data(x, file, files) 

main()