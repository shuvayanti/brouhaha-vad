from scipy.io import wavfile
from scipy import stats
import numpy as np
import glob
import csv
import yaml
import librosa
import os
import math

# calculatee the RMS of an audio clip based on onset and duration
def mean_RMS(filename, onset, duration):

    # load the audio using librosa 
    y, sr = librosa.load(f'/scratch2/mlavechin/VoiceTypeClassifierPaper/DATA/BabyTrain_full_resplitted2/test/{filename}.wav')

    # calculate the onset frame
    onset_frame = math.ceil(onset * sr)

    #calculate the number of frames based on duration of the clip
    n_frames = math.ceil(duration * sr)

    # calculate the offset frame
    offset_frame = onset_frame + n_frames

    return onset_frame, offset_frame, n_frames, np.sqrt(np.mean(y[onset_frame:offset_frame+1]**2))      # return the RMS of the audio clip from onset-offset

def rms_vtc_labels():

    # create csv files to store the results of RMS of kCHI & CHI clips in VTC labelled audio files.
    write_file_kchi = csv.writer(open('/home/sdas/brouhaha-vad/analysis/vtc-rms-kchi-labels.csv','w',newline = '\n'))
    write_file_ochi = csv.writer(open('/home/sdas/brouhaha-vad/analysis/vtc-rms-ochi-labels.csv','w',newline = '\n'))

    write_file_kchi.writerow(['Name', 'VTC label', 'Onset(s)', 'Duration(s)', 'Onset(frame)', 'Offset(frame)', 'No. of frames', 'Mean RMS'])
    write_file_ochi.writerow(['Name', 'VTC label', 'Onset(s)', 'Duration(s)', 'Onset(frame)', 'Offset(frame)', 'No. of frames', 'Mean RMS'])

    # Process each file in the VTC directory
    for file in glob.glob('/home/sdas/pyannote-vtc-testing/runs/babytrain_2/apply/*.rttm'):
        print('\n',file)
        
        lines = open(file).read().splitlines()

        for line in lines:
            line = line.split(' ')
            label = line[7]
            filename = line[1].split('/')[-1]
            onset = line[3]
            duration = line[4]
            
            if label == 'KCHI':
                onset_frame, offset_frame, n_frames, mean_rms = mean_RMS(filename , float(onset), float(duration))
                write_file_kchi.writerow([filename, label, onset, duration, onset_frame, offset_frame, n_frames, mean_rms])

            elif label == 'CHI':
                onset_frame, offset_frame, n_frames, mean_rms = mean_RMS(filename , float(onset), float(duration))
                write_file_ochi.writerow([filename, label, onset, duration, onset_frame, offset_frame, n_frames, mean_rms])
            

#function to find and write the RMS of kCHI & CHI clips to csv of human annotated audio clips.
def rms_gold_labels():

    # load human-to-label mapping
    mapping = yaml.safe_load(open('/home/sdas/pyannote-vtc-testing/data/babytrain_mapping.yml'))['mapping']
    mapping['MOT']='FEM'
    mapping['FAT']='MAL'

    write_file_kchi = csv.writer(open('/home/sdas/brouhaha-vad/analysis/gold-rms-kchi-labels.csv','w',newline = '\n'))
    write_file_ochi = csv.writer(open('/home/sdas/brouhaha-vad/analysis/gold-rms-ochi-labels.csv','w',newline = '\n'))

    write_file_kchi.writerow(['Name', 'Gold label', 'Onset(s)', 'Duration(s)', 'Onset(frame)', 'Offset(frame)', 'No. of frames', 'Mean RMS'])
    write_file_ochi.writerow(['Name', 'Gold label', 'Onset(s)', 'Duration(s)', 'Onset(frame)', 'Offset(frame)', 'No. of frames', 'Mean RMS'])

    # Process each human annotated file in the test folder.
    for file in glob.glob('/scratch2/mlavechin/VoiceTypeClassifierPaper/DATA/BabyTrain_full_resplitted2/test/*.test.rttm')[:-1]:
        lines = open(file).read().splitlines()

        for line in lines:
            line = line.split(' ')
            annotation = line[7]
            try:
                label = mapping[annotation]
            except:
                label = annotation

            filename = line[1].split('/')[-1]
            onset = line[3]
            duration = line[4]

            if label == 'KCHI':
                onset_frame, offset_frame, n_frames, mean_rms = mean_RMS(filename , float(onset), float(duration))
                write_file_kchi.writerow([filename, f'{label}({annotation})', onset, duration, onset_frame, offset_frame, n_frames, mean_rms])
            elif label == 'CHI':
                onset_frame, offset_frame, n_frames, mean_rms = mean_RMS(filename , float(onset), float(duration))
                write_file_ochi.writerow([filename, f'{label}({annotation})', onset, duration, onset_frame, offset_frame, n_frames, mean_rms])



if __name__ == '__main__':

    os.system('rm -r /home/sdas/brouhaha-vad/analysis/*rms*.csv')

    # Run the codes
    rms_vtc_labels()
    #rms_gold_labels()