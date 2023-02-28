import numpy as np
import glob
import csv
import math
import yaml
import os

#calculate the number of dimensions( SNR numpy array) per second
def calculate_dimensions():
    
    filename = open('/scratch2/mlavechin/VoiceTypeClassifierPaper/DATA/BabyTrain_full_resplitted2/test/aclew.test.uem').read().splitlines()[0].split(' ')[0].split('/')[-1]

    duration = float(open('/scratch2/mlavechin/VoiceTypeClassifierPaper/DATA/BabyTrain_full_resplitted2/test/aclew.test.uem').read().splitlines()[0].split(' ')[-1])

    dimensions = len(np.load(f'/home/sdas/brouhaha-vad/predictions/detailed_snr_labels/{filename}.npy'))

    return dimensions/duration

# the following function calculates the mean SNR of an audio clip
def mean_SNR(filename, onset, duration, n_dimension_per_sec):

    # load the SNR file
    detailed_snr = np.load(f'/home/sdas/brouhaha-vad/predictions/detailed_snr_labels/{filename}.npy')

    # calculate the onset dimension of the SNR numpy array
    onset_dim = math.ceil(onset * n_dimension_per_sec)

    # number of dimensions of the numpy array corresponding to duration of the clip
    n_dim = math.ceil(duration * n_dimension_per_sec)

    # calculate the offset dimension
    offset_dim = onset_dim + n_dim

    return onset_dim, offset_dim, n_dim, np.mean(detailed_snr[onset_dim:offset_dim])       #return the mean SNR from the onset-offset

#function to find and write the SNR of kCHI & CHI clips to csv of VTC annotated audio clips.
def snr_vtc_labels(n_dimension_per_sec):

    # create files to store SNR of VTC labelled audio clips of kCHI & CHI
    write_file_kchi = csv.writer(open('vtc-snr-kchi-labels.csv','w',newline = '\n'))
    write_file_ochi = csv.writer(open('vtc-snr-ochi-labels.csv','w',newline = '\n'))

    write_file_kchi.writerow(['Name', 'VTC label', 'Onset(s)', 'Duration(s)', 'Onset(dimension)', 'Offset(dimension)', 'No. of dimensions', 'Mean SNR'])
    write_file_ochi.writerow(['Name', 'VTC label', 'Onset(s)', 'Duration(s)', 'Onset(dimension)', 'Offset(dimension)', 'No. of dimensions', 'Mean SNR'])

    # Process the VTC labelled files in the test folder
    for file in glob.glob('/home/sdas/pyannote-vtc-testing/runs/babytrain_2/apply/*.rttm'):
        
        lines = open(file).read().splitlines()

        for line in lines:
            line = line.split(' ')
            label = line[7]
            filename = line[1].split('/')[-1]
            onset = line[3]
            duration = line[4]
            
            if label == 'KCHI':
                onset_dim, offset_dim, n_dim, mean_snr = mean_SNR(filename , float(onset), float(duration), n_dimension_per_sec)
                write_file_kchi.writerow([filename, label, onset, duration, onset_dim, offset_dim, n_dim, mean_snr])
            elif label == 'CHI':
                onset_dim, offset_dim, n_dim, mean_snr = mean_SNR(filename , float(onset), float(duration), n_dimension_per_sec)
                write_file_ochi.writerow([filename, label, onset, duration, onset_dim, offset_dim, n_dim, mean_snr])

#function to find and write the SNR of kCHI & CHI clips to csv of human annotated audio clips.
def snr_gold_labels(n_dimension_per_sec):

    # Load the human-to-label mapping
    mapping = yaml.safe_load(open('/home/sdas/pyannote-vtc-testing/data/babytrain_mapping.yml'))['mapping']
    mapping['MOT']='FEM'
    mapping['FAT']='MAL'

    write_file_kchi = csv.writer(open('gold-snr-kchi-labels.csv','w',newline = '\n'))
    write_file_ochi = csv.writer(open('gold-snr-ochi-labels.csv','w',newline = '\n'))

    write_file_kchi.writerow(['Name', 'Gold label', 'Onset(s)', 'Duration(s)', 'Onset(dimension)', 'Offset(dimension)', 'No. of dimensions', 'Mean SNR'])
    write_file_ochi.writerow(['Name', 'Gold label', 'Onset(s)', 'Duration(s)', 'Onset(dimension)', 'Offset(dimension)', 'No. of dimensions', 'Mean SNR'])


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
                onset_dim, offset_dim, n_dim, mean_snr = mean_SNR(filename , float(onset), float(duration), n_dimension_per_sec)
                write_file_kchi.writerow([filename, f'{label}({annotation})', onset, duration, onset_dim, offset_dim, n_dim, mean_snr])
            elif label == 'CHI':
                onset_dim, offset_dim, n_dim, mean_snr = mean_SNR(filename , float(onset), float(duration), n_dimension_per_sec)
                write_file_ochi.writerow([filename, f'{label}({annotation})', onset, duration, onset_dim, offset_dim, n_dim, mean_snr])



if __name__ == '__main__':
    n_dimension_per_sec = calculate_dimensions()

    print(n_dimension_per_sec)

    # remove any preexisting files to prevent contamination
    os.system('rm -r /home/sdas/brouhaha-vad/analysis/*snr*.csv')

    snr_vtc_labels(n_dimension_per_sec)
    snr_gold_labels(n_dimension_per_sec)