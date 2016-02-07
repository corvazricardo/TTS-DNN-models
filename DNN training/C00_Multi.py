# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

# This code builds the multi-speaker speech corpus from the VCTK corpus.
# Speech recordings are set to use 5 British and 5 scottish accent speakers.
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
    
    def __init__(self, n_ins=601, n_outs=259, n_speakers_accent=1,speaker_accent = 'Scottish',shuffle=False):
        
        
        try:
            assert n_speakers_accent > 0 and n_speakers_accent <= 5
        except AssertionError:
            print 'Number of speakers per accent allowed is from 1 up to 5 '
            raise
        
        
        
        
        #Setting Initial Values for Initial List
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
        self.speaker_accent = speaker_accent
    
        self.list_size = 0
        self.n_speakers = n_speakers_accent
        
        
        
        #Getting input/output file lists
        self.file_list(self.speaker_accent,'input')
        self.file_list(self.speaker_accent,'output')
        
    
    def file_list(self,speaker_accent,input_output_mode):
        

		
	# Speakers Corpus for the dissertation project	

	english_accent = ['p389_r356_20120409', 'p337_r313_20120311', 'p291_r247_20120208', 'p273_r414_20120423', 'p226_r367_20120423' ]
        
        scottish_accent = ['p395_r364_20120418','p397_r430_20120423', 'p46_r48_20110919', 'p327_r286_20120227', 'p114_r116_20111001' ]
        


        
        s_folder_list = []
        
        if speaker_accent == 'Female' or speaker_accent =='Male':

            type_accent_1 = 'Female'
            type_accent_2 = 'Male'
            accent_1 = female_accent
            accent_2 = male_accent
            comment_1 = "Obtaining Female List"
            comment_2 = "Obtaining Male List"
            
        elif speaker_accent == 'English' or speaker_accent =='Scottish':
            
            type_accent_1 = 'English'
            type_accent_2 = 'Scottish'
            accent_1 = english_accent
            accent_2 = scottish_accent
            comment_1 = "Obtaining English List"
            comment_2 = "Obtaining Scottish List"

	# Access to EDDIE server might be required. 
	# We include username and password to ensure the access to the EDDIE server
        
        sftp = pysftp.Connection('eddie.ecdf.ed.ac.uk',username='myusername',password='mypassword')
        
        #Setting path for reading Corpus    
	# Access to path of EDDIE server might be required
        if input_output_mode == 'input': 

            path= "/exports/work/inf_hcrc_cstr_nst/zhizheng/DNN/share_ricardo/nn_no_silence_lab_norm_601"
            sftp.chdir(path)
            speakers_folder_list = sftp.listdir()
            
            if speaker_accent == type_accent_1:
                
                print comment_1," :", input_output_mode
                
                for speaker in speakers_folder_list:
                    if speaker in accent_1:

                        s_folder_list.append(speaker)
                            
                
            elif speaker_accent == type_accent_2:
                
                print comment_2," :",input_output_mode
                
                for speaker in speakers_folder_list:
                    if speaker in accent_2:

                        s_folder_list.append(speaker)


                    
            self.i_speakers_list = s_folder_list[0:self.n_speakers]
            #print "Lenght of speakers list in C00_Multi:",self.i_speakers_list
            speakers_list = self.i_speakers_list
            
        elif input_output_mode == 'output':

            path= "/exports/work/inf_hcrc_cstr_nst/zhizheng/DNN/share_ricardo/nn_norm_mgc_lf0_vuv_bap_259"
            sftp.chdir(path)
            speakers_folder_list = sftp.listdir()
            
            if speaker_accent == type_accent_1:
                
                print comment_1," :",input_output_mode
                
                
                for speaker in speakers_folder_list:
                    if speaker in accent_1:

                        s_folder_list.append(speaker)
                            
                
            elif speaker_accent == type_accent_2:
                
                print comment_2," :",input_output_mode
                
                for speaker in speakers_folder_list:
                    if speaker in accent_2:

                        s_folder_list.append(speaker)
            
            
            self.o_speakers_list = s_folder_list[0:self.n_speakers]
            speakers_list = self.o_speakers_list
        

        file_list = []
        frame_number = []
        self.speaker_list = [0]


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
        

