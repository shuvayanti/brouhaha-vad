import csv
import numpy as np
import os
import glob
import math

def calculate_dimensions():
    
    filename = open('/scratch2/mlavechin/VoiceTypeClassifierPaper/DATA/BabyTrain_full_resplitted2/test/aclew.test.uem').read().splitlines()[0].split(' ')[0].split('/')[-1]

    print(filename)

    duration = float(open('/scratch2/mlavechin/VoiceTypeClassifierPaper/DATA/BabyTrain_full_resplitted2/test/aclew.test.uem').read().splitlines()[0].split(' ')[-1])

    dimensions = len(np.load(f'../predictions/detailed_RMS/{filename}.npy'))

    global unit_dim

    unit_dim = dimensions/duration

def squash_segments_vtc():
    
    os.makedirs('segments/VTC/',exist_ok=True)

    files = glob.glob('../../pyannote-vtc-testing/runs/babytrain_2/apply/*.rttm')

    write_for_all_files = csv.writer(open('segments/mean_rms_non_speech_vtc.csv','w',newline='\n'))
    write_for_all_files.writerow(['Filename','Mean RMS(Non-Speech)'])

    for f in files:
        filename = f.split('/')[-1].split('.')[0][5:]

        csv_write = csv.writer(open(f'segments/VTC/{filename}.csv','w',newline = '\n'))

        current_onset = 0 
        current_offset = 0
        count = 0

        rms = np.load(f'../predictions/detailed_RMS/{filename}.npy')
        #print(rms)
        lines = open(f).read().splitlines()
        total_rms = 0
        for line in lines:

            line = line.split(' ')

            new_onset = float(line[3])
            duration = float(line[4])

            current_offset_dim = math.ceil(current_offset * unit_dim)
            new_onset_dim = math.floor(new_onset * unit_dim)

            if new_onset_dim > current_offset_dim:

                mean_rms = np.mean(rms[current_offset_dim:new_onset_dim])
                
                total_rms += mean_rms
                csv_write.writerow([current_offset, new_onset, new_onset-current_offset, 'NON-SPEECH',mean_rms])
                current_onset = new_onset
                current_offset = current_onset + duration
                count+=1
            
            elif new_onset < current_offset and new_onset+duration > current_offset:
                current_offset = new_onset+duration
        if count >0:
            write_for_all_files.writerow([filename,total_rms/count])

def squash_segments_human():
    os.makedirs('segments/human/',exist_ok=True)

    files = glob.glob('/scratch2/mlavechin/VoiceTypeClassifierPaper/DATA/BabyTrain_full_resplitted2/test/*.test.rttm')[:-1]

    write_for_all_files = csv.writer(open('segments/mean_rms_non_speech_human.csv','w',newline='\n'))
    write_for_all_files.writerow(['Filename','Mean RMS(Non-Speech)'])

    old_file = ''
    count = 0
    current_onset = 0 
    current_offset = 0
    total_rms =0

    for f in files:
        lines = open(f).read().splitlines()

        for line in lines:
            line = line.split(' ')

            new_file = line[1].split('/')[-1]

            rms = np.load(f'../predictions/detailed_RMS/{new_file}.npy')

            if new_file != old_file:

                if count >0:
                    #print('When writing:',old_file, total_rms, count)
                    write_for_all_files.writerow([old_file,total_rms/count])

                csv_write = csv.writer(open(f'segments/human/{new_file}.csv','w',newline = '\n'))
                old_file =new_file
                count = 0

                current_onset = 0 
                current_offset = 0

                total_rms = 0
            
            new_onset = float(line[3])
            duration = float(line[4])

            current_offset_dim = math.ceil(current_offset * unit_dim)
            new_onset_dim = math.floor(new_onset * unit_dim)

            if new_onset_dim > current_offset_dim:
                
                #print('RMS: ',rms[current_offset_dim:new_onset_dim])
                
                mean_rms = np.mean(rms[current_offset_dim:new_onset_dim])
                #print('Mean RMS:', mean_rms)
                total_rms += mean_rms
                csv_write.writerow([current_offset, new_onset, new_onset-current_offset, 'NON-SPEECH',mean_rms])
                current_onset = new_onset
                current_offset = current_onset + duration
                count+=1
            
            elif new_onset < current_offset and new_onset+duration > current_offset:
                current_offset = new_onset+duration

if __name__ == '__main__':
    
    os.system('rm -r segments')
    calculate_dimensions()
    print('Unit dim: ',unit_dim)
    squash_segments_vtc()
    squash_segments_human()