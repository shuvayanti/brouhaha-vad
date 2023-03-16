from scipy.io import wavfile
from scipy import stats
import numpy as np
import glob
import csv
import yaml
import librosa
import os
import math
import pandas as pd

# calculatee the RMS of an audio clip based on onset and duration
def calculate_RMS():

    os.makedirs('../predictions/detailed_RMS/',exist_ok=True)
    for wavfile in glob.glob('/scratch2/mlavechin/VoiceTypeClassifierPaper/DATA/BabyTrain_full_resplitted2/test/*.wav'):

        filename = wavfile.split('/')[-1].split('.')[0]
        y, sr = librosa.load(wavfile)

        np.save(f'../predictions/detailed_RMS/{filename}.npy',librosa.feature.rms(y=y, frame_length = 1600, hop_length = 1600)[0])

#calculate the number of dimensions( SNR numpy array) per second
def calculate_dimensions():
    
    filename = open('/scratch2/mlavechin/VoiceTypeClassifierPaper/DATA/BabyTrain_full_resplitted2/test/aclew.test.uem').read().splitlines()[0].split(' ')[0].split('/')[-1]

    print(filename)

    duration = float(open('/scratch2/mlavechin/VoiceTypeClassifierPaper/DATA/BabyTrain_full_resplitted2/test/aclew.test.uem').read().splitlines()[0].split(' ')[-1])

    dimensions = len(np.load(f'../predictions/detailed_RMS/{filename}.npy'))

    global n_dimension_per_sec

    n_dimension_per_sec = dimensions/duration
    
def rms_vtc_labels():

    # create csv files to store the results of RMS of kCHI & CHI clips in VTC labelled audio files.
    write_file_kchi = csv.writer(open('RMS/vtc-rms-kchi(signal_only)-labels.csv','w',newline = '\n'))
    write_file_ochi = csv.writer(open('RMS/vtc-rms-ochi(signal_only)-labels.csv','w',newline = '\n'))

    write_file_kchi.writerow(['Name', 'VTC label', 'Onset(s)', 'Duration(s)', 'Onset(dim)', 'Offset(dim)', 'No. of dimensions', 'Mean RMS(Speech only)'])
    write_file_ochi.writerow(['Name', 'VTC label', 'Onset(s)', 'Duration(s)', 'Onset(dim)', 'Offset(dim)', 'No. of dimensions', 'Mean RMS(Speech only)'])

    rms_non_speech = pd.read_csv('segments/mean_rms_non_speech_vtc.csv')

    # Process each file in the VTC directory
    for file in glob.glob('../../pyannote-vtc-testing/runs/babytrain_2/apply/*.rttm'):
        print('\n',file)
        
        filename = file.split('/')[-1].split('.')[0][5:]

        print('\nProcessing: ',filename)

        rms = np.load(f'../predictions/detailed_RMS/{filename}.npy')
        #print('\nBefore: ',rms_non_speech[rms_non_speech['Filename']==filename]['Mean RMS(Non-Speech)'])
        try:
            mean_rms_non_speech = rms_non_speech[rms_non_speech['Filename']==filename]['Mean RMS(Non-Speech)'].item()
        except:
            mean_rms_non_speech = 0
        #print('After: ',mean_rms_non_speech,'\n')
        lines = open(file).read().splitlines()

        for line in lines:
            line = line.split(' ')
            label = line[7]
            
            onset = float(line[3])
            duration = float(line[4])

            
            
            if label == 'KCHI':

                # calculate the onset dimension
                onset_dim = math.ceil(onset * n_dimension_per_sec)

                #calculate the number of dimensions based on the array length
                n_dim = math.ceil(duration * n_dimension_per_sec)

                # calculate the offset dimension
                offset_dim = onset_dim + n_dim

                # extract the SNR in dB 
                mean_rms = math.log( np.mean(rms[onset_dim:offset_dim])**2 / mean_rms_non_speech**2, 10)
                #print(mean_rms)
                write_file_kchi.writerow([filename, label, onset, duration, onset_dim, offset_dim, n_dim, mean_rms])

            elif label == 'CHI':

                onset_dim = math.ceil(onset * n_dimension_per_sec)

                n_dim = math.ceil(duration * n_dimension_per_sec)

                # calculate the offset dimension
                offset_dim = onset_dim + n_dim

                mean_rms = math.log( np.mean(rms[onset_dim:offset_dim])**2 / mean_rms_non_speech**2, 10)
                #print(mean_rms)
                write_file_ochi.writerow([filename, label, onset, duration, onset_dim, offset_dim, n_dim, mean_rms])

            

#function to find and write the RMS of kCHI & CHI clips to csv of human annotated audio clips.
def rms_gold_labels():

    # load human-to-label mapping
    mapping = yaml.safe_load(open('../../pyannote-vtc-testing/data/babytrain_mapping.yml'))['mapping']
    mapping['MOT']='FEM'
    mapping['FAT']='MAL'

    write_file_kchi = csv.writer(open('RMS/gold-rms-kchi(signal_only)-labels.csv','w',newline = '\n'))
    write_file_ochi = csv.writer(open('RMS/gold-rms-ochi(signal_only)-labels.csv','w',newline = '\n'))

    write_file_kchi.writerow(['Name', 'Gold label', 'Onset(s)', 'Duration(s)', 'Onset(dim)', 'Offset(dim)', 'No. of dimensions', 'Mean RMS(Speech only)'])
    write_file_ochi.writerow(['Name', 'Gold label', 'Onset(s)', 'Duration(s)', 'Onset(dim)', 'Offset(dim)', 'No. of dimensions', 'Mean RMS(Speech only)'])

    rms_non_speech = pd.read_csv('segments/mean_rms_non_speech_human.csv')

    files = glob.glob('/scratch2/mlavechin/VoiceTypeClassifierPaper/DATA/BabyTrain_full_resplitted2/test/*.test.rttm')[:-1]

    # Process each human annotated file in the test folder.
    for file in files:

        print('\n',file)
        
        lines = open(file).read().splitlines()

        for line in lines:
            line = line.split(' ')
            annotation = line[7]

            try:
                label = mapping[annotation]
            except:
                label = annotation

            filename = line[1].split('/')[-1]
            onset = float(line[3])
            duration = float(line[4])

            print('Processing: ',filename)

            rms = np.load(f'../predictions/detailed_RMS/{filename}.npy')
            try:
                mean_rms_non_speech = rms_non_speech[rms_non_speech['Filename']==filename]['Mean RMS(Non-Speech)'].item()
            except:
                mean_rms_non_speech = 0
            #print(mean_rms_non_speech)

            if label == 'KCHI':
                onset_dim = math.ceil(onset * n_dimension_per_sec)

                n_dim = math.ceil(duration * n_dimension_per_sec)

                # calculate the offset dimension
                offset_dim = onset_dim + n_dim

                mean_rms = math.log( np.mean(rms[onset_dim:offset_dim])**2 / mean_rms_non_speech**2, 10)
                #print('Mean RMS(with noise), RMS (Noise), Mean RMS(w/o noise):',np.mean(rms[onset_dim:offset_dim]), mean_rms_non_speech, mean_rms)
                write_file_kchi.writerow([filename, f'{label}({annotation})', onset, duration, onset_dim, offset_dim, n_dim, mean_rms])

            elif label == 'CHI':
                onset_dim = math.ceil(onset * n_dimension_per_sec)

                n_dim = math.ceil(duration * n_dimension_per_sec)

                # calculate the offset dimension
                offset_dim = onset_dim + n_dim

                mean_rms = math.log( np.mean(rms[onset_dim:offset_dim])**2 / mean_rms_non_speech**2, 10)
                #print('Mean RMS(with noise), RMS (Noise), Mean RMS(w/o noise):',np.mean(rms[onset_dim:offset_dim]), mean_rms_non_speech, mean_rms)
                write_file_ochi.writerow([filename, f'{label}({annotation})', onset, duration, onset_dim, offset_dim, n_dim, mean_rms])

if __name__ == '__main__':

    calculate_dimensions()

    os.system('rm -r RMS/*signal_only*.csv')

    # Run the codes
    rms_vtc_labels()
    rms_gold_labels()


    #calculate_RMS()
    
