# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>


# This code load input features of each speaker to train the DNN model.
# Author: R. Cortez
# The University of Edinburgh
# Version: Aug 19 2015

#import numpy, theano, random
from binary_io import BinaryIOCollection

#import binary_io
import os
import os.path
import pysftp

# <rawcell>

# CREATE INITIAL LIST

# <codecell>

class Initial_list(object):
    
    def __init__(self, n_ins=601, n_outs=259, n_speakers=1,speaker_accent = 'Female',shuffle=False):
        
        
        try:
            assert n_speakers > 0 and n_speakers <= 5
        except AssertionError:
            print 'Number of speakersallowed is from 1 up to 5 '
            raise
        
        
        self.n_ins = n_ins
        self.n_outs = n_outs
        
        self.i_list = []
        self.o_list = []
        
        self.i_speaker_list = []
        self.o_speaker_list = []
        self.speakers_list = []
        self.speaker_ID = []
        speakers_list = []
        speaker_list = []
        
        self.list_size = 0
        self.n_speakers = n_speakers
        self.speaker_accent = speaker_accent
        
        
        
        self.file_list('input',self.speaker_accent)
        self.file_list('output',self.speaker_accent)
    
    def file_list(self,input_output_mode,speaker_accent):
        

        
        english_accent = ['p389_r356_20120409', 'p337_r313_20120311', 'p291_r247_20120208', 'p273_r414_20120423', 'p226_r367_20120423' ]
        
        scottish_accent = ['p395_r364_20120418','p397_r430_20120423', 'p46_r48_20110919', 'p327_r286_20120227', 'p114_r116_20111001' ]
        
        

        
        s_folder_list = []
        
        sftp = pysftp.Connection('eddie.ecdf.ed.ac.uk',username='myusername',password='mypassword')
        
        type_accent = self.speaker_accent
        
        if type_accent == 'Female':
            accent = female_accent
        elif type_accent == 'Male':
            accent = male_accent
        elif type_accent == 'Scottish':
            accent = scottish_accent
        elif type_accent == 'English':
            accent = english_accent
        
        #Setting path for reading Corpus    
        if input_output_mode == 'input': 

            path= "/exports/work/inf_hcrc_cstr_nst/zhizheng/DNN/share_ricardo/nn_no_silence_lab_norm_601"
            sftp.chdir(path)
            speakers_folder_list = sftp.listdir() 
            
            for speaker in speakers_folder_list:
                if speaker in accent:
                    s_folder_list.append(speaker)
            
            self.i_speakers_list = s_folder_list[0:self.n_speakers]
            speakers_list = self.i_speakers_list
        elif input_output_mode == 'output':
            path= "/exports/work/inf_hcrc_cstr_nst/zhizheng/DNN/share_ricardo/nn_norm_mgc_lf0_vuv_bap_259"
            sftp.chdir(path)
            speakers_folder_list = sftp.listdir()
            
            for speaker in speakers_folder_list:
                if speaker in accent:
                    s_folder_list.append(speaker)
            
            self.o_speakers_list = s_folder_list[0:self.n_speakers]
            speakers_list = self.o_speakers_list
        
        #List of speakers data available

        file_list = []
        frame_number = []
        self.speaker_list = [0]* len(self.i_speakers_list) 


        #Load input features of each speaker in a List
        for folder in speakers_list:
            sftp.chdir(path + '/' + folder)
            input_speaker_recording_list = sftp.listdir()
            speaker_list = input_speaker_recording_list
            if input_output_mode == 'input':
                self.speaker_ID.append(len(input_speaker_recording_list))
            
            for recording in speaker_list:
                file_name = path + '/' + folder + '/' + recording
                file_list.append(file_name)
        self.end_reading = False  
        
        if input_output_mode == 'input': 
            self.i_list = file_list
            
        elif input_output_mode == 'output':
            self.o_list = file_list
        
        sftp.close()
        

# <codecell>


# <codecell>


# <codecell>


