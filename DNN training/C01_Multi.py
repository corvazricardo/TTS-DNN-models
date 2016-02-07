# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>
# Part of the code was used from code of Dr. Zhizheng Wu.
# Code edited and extended by R. Cortez
# Author: Dr. Zhizheng Wu and R. Cortez
# The University of Edinburgh
# Version: Aug 19 2015


#import numpy, theano, random
from binary_io import BinaryIOCollection

#import binary_io
import os
import os.path

#  numpy & theano imports need to be done in this order (only for some numpy installations, not sure why)
import numpy
# we need to explicitly import this in some cases, not sure why this doesn't get imported with numpy itself
import numpy.distutils.__config__
# and only after that can we import theano 
import theano
import theano.tensor as T
import random

from C00_Multi import Initial_list
import pysftp


theano.config.floatX = 'float32'

# <rawcell>

# CREATE VCTK COLLECTION OBJECT

# <codecell>

class VCTK_feat_collection(object):
    
    def __init__(self, i_list=[] , o_list=[], n_ins=601, n_outs=259, n_speakers=1, buffer_size = 500000, shuffle=False):
        
        print ("Initialising VCTK Features Collection")
        
        self.n_ins = n_ins
        self.n_outs = n_outs
        self.buffer_size = buffer_size
        
        self.i_list = i_list
        self.o_list = o_list
        
        self.speaker_list = []
        self.speakers_list = []
        self.remain_frame_number = 0 
        self.list_size = 0
        self.n_speakers = n_speakers
        
        self.end_reading = False
        
        
        try:
            assert len(self.i_list) > 0
        except AssertionError:
            print 'Input data list is empty'
            raise

        try:
            assert len(self.o_list) > 0
        except AssertionError:
            print 'Output data list is empty'
            raise

        try:
            assert len(self.i_list) == len(self.o_list)
        except AssertionError:
            print 'Two lists have different lengths :', len(self.i_list), 'versus', len(self.o_list)
            raise
            
        print 'First  list of items from : ', self.i_list[0].rjust(20)[-20:], '  to  ', self.i_list[-1].rjust(20)[-20:]
        print 'Second  list of items from : ', self.o_list[0].rjust(20)[-20:], '  to  ', self.o_list[-1].rjust(20)[-20:]

        if shuffle: 
            random.seed(271638)
            random.shuffle(self.i_list)
            random.seed(271638)
            random.shuffle(self.o_list)

        self.file_index = 0
        self.list_size = len(self.i_list)
        
        self.i_remain_data = numpy.empty((0, self.n_ins))
        self.o_remain_data = numpy.empty((0, self.n_outs))
        self.remain_frame_number = 0

        self.end_reading = False
        
        print "Initialised"
    

        
    

# <rawcell>

# RESET OBJECT VALUES

# <codecell>

    def reset(self):
        self.file_index = 0
        self.end_reading = False
        self.remain_frame_number = 0
        print 'Reset Object Values'

# <rawcell>

# ITERANCE

# <codecell>

    def __iter__(self):
        return self

# <rawcell>

# MAKE DATA SHARED

# <codecell>

    def make_shared(self, data_set, data_name):
        data_set = theano.shared(numpy.asarray(data_set, dtype=theano.config.floatX), name=data_name, borrow=True)
        return  data_set
    

# <rawcell>

# GET FILE LIST DATA "PROBABLY WE DON'T NEED THIS METHOD"

# <codecell>

    def get_list(self,input_output_mode):
        if input_output_mode == 'input': 
            file_list = self.i_list
            
        elif input_output_mode == 'output':
            file_list = self.o_list
        return file_list

# <rawcell>

# LOAD NEXT UTTERANCE

# <codecell>

    def load_next_utterance(self):
        
        print ("Loading Next Utterance")

        
        i_temp_set = numpy.empty((self.buffer_size, self.n_ins))
        o_temp_set = numpy.empty((self.buffer_size, self.n_outs))

        io_function = BinaryIOCollection()
        sftp = pysftp.Connection('eddie.ecdf.ed.ac.uk',username='s1459795',password='Edinburgh14')
         
        sftp.get(self.i_list[self.file_index])
        sftp.get(self.o_list[self.file_index])
        
        i_file_name = self.extract_file_id(self.i_list[self.file_index])
        o_file_name = self.extract_file_id(self.o_list[self.file_index])
        
        remove_items = []
        
        while (os.path.isfile(i_file_name) and os.path.isfile(o_file_name)) is False:
            
            print i_file_name," is :", os.path.isfile(i_file_name), "in load_next_utterance"
            print o_file_name," is :", os.path.isfile(o_file_name), "in load_next_utterance"
            
            remove_items.append(i_file_name)
            remove_items.append(o_file_name)
            
            self.file_index +=1
            
            sftp.get(self.i_list[self.file_index])
            sftp.get(self.o_list[self.file_index])
            
            i_file_name = self.extract_file_id(self.i_list[self.file_index])
            o_file_name = self.extract_file_id(self.o_list[self.file_index])
        
        
        remove_items.append(i_file_name)
        remove_items.append(o_file_name)
        sftp.close()
        
        i_features, lab_frame_number = io_function.load_binary_file_frame(i_file_name, self.n_ins)
        o_features, out_frame_number = io_function.load_binary_file_frame(o_file_name, self.n_outs)
        
        for item in remove_items:
            if os.path.isfile(item) is True:
                os.remove(item)

        remove_items = []
        
        frame_number = lab_frame_number
        
        if abs(lab_frame_number - out_frame_number) < 5:    ## we allow small difference here. may not be correct, but sometimes, there is one/two frames difference
            if lab_frame_number > out_frame_number:
                frame_number = out_frame_number
        else:
            print "The number of frames in label and acoustic features are different  : ",lab_frame_number, " vs " , out_frame_number
            raise
            
        i_temp_set = o_features[0:frame_number, ]
        o_temp_set = i_features[0:frame_number, ]

        self.file_index += 1
        
        
        if  self.file_index >= self.list_size:
            self.end_reading = True
            self.file_index = 0

        i_shared_set = self.make_shared(i_temp_set, 'x')
        o_shared_set = self.make_shared(o_temp_set, 'y')

        io_shared_set = (i_shared_set, o_shared_set)

        return io_shared_set, i_temp_set, o_temp_set

# <rawcell>

# LOAD NEXT PARTITION

# <codecell>

    def load_next_partition(self):
        
        print 'Loading Next Partition'
        
        sftp = pysftp.Connection('eddie.ecdf.ed.ac.uk',username='s1459795',password='Edinburgh14')
        
        i_temp_set = numpy.empty((self.buffer_size, self.n_ins))
        o_temp_set = numpy.empty((self.buffer_size, self.n_outs))
        current_index = 0

        ### first check whether there are remaining data from previous utterance
        if self.remain_frame_number > 0:
            
            remain_frames = self.remain_frame_number - current_index
            remain_size = self.i_remain_data.shape[0]
            
            if remain_frames != remain_size:
                
                if remain_frames < remain_size:
                    
                    if remain_size - remain_frames <= 20:
                        print "Check remain_size"
                    else:
                        
                        print "We delete", self.remain_frame_number,"frames"
                        self.remain_frame_number = 0
                        
                elif remain_frames > remain_size:
                    self.remain_frame_number = self.remain_frame_number - (remain_frames - remain_size)
            
            i_temp_set[current_index:self.remain_frame_number, ] = self.i_remain_data
            o_temp_set[current_index:self.remain_frame_number, ] = self.o_remain_data
            current_index += self.remain_frame_number
            
            self.remain_frame_number = 0
        
        io_function = BinaryIOCollection()  
        while True:
            if current_index >= self.buffer_size:
                break
            if  self.file_index >= self.list_size:
                self.end_reading = True
                self.file_index = 0
                break
               
 
            sftp.get(self.i_list[self.file_index])
            sftp.get(self.o_list[self.file_index])
            
            
            i_file_name = self.extract_file_id(self.i_list[self.file_index])
            o_file_name = self.extract_file_id(self.o_list[self.file_index])
            
            remove_items = []
        
            while (os.path.isfile(i_file_name) and os.path.isfile(o_file_name)) is False:
                
                print i_file_name," is :", os.path.isfile(i_file_name), "in beginning of load_next_partition"
                print o_file_name," is :", os.path.isfile(o_file_name), "in beginning of load_next_partition"
            
                remove_items.append(i_file_name)
                remove_items.append(o_file_name)
            
                self.file_index +=1
            
                sftp.get(self.i_list[self.file_index])
                sftp.get(self.o_list[self.file_index])
            
                i_file_name = self.extract_file_id(self.i_list[self.file_index])
                o_file_name = self.extract_file_id(self.o_list[self.file_index])
            
            remove_items.append(i_file_name)
            remove_items.append(o_file_name)
            

            i_features, lab_frame_number = io_function.load_binary_file_frame(i_file_name , self.n_ins)    
            o_features, out_frame_number = io_function.load_binary_file_frame(o_file_name, self.n_outs)
           
            for item in remove_items:
                if os.path.isfile(item) is True:
                    os.remove(item)

            remove_items = []
            

            frame_number = lab_frame_number
            if abs(lab_frame_number - out_frame_number) < 5:    ## we allow small difference here. may not be correct, but sometimes, there is one/two frames difference
                if lab_frame_number > out_frame_number:
                    frame_number = out_frame_number
            else:
                while abs(lab_frame_number - out_frame_number) > 5:
                    self.file_index += 1
                    
                    #############
                    while (os.path.isfile(i_file_name) and os.path.isfile(o_file_name)) is False:
                
                        print i_file_name," is :", os.path.isfile(i_file_name), "in ELSE of load_next_partition"
                        print o_file_name," is :", os.path.isfile(o_file_name), "in ELSE of load_next_partition"
            
                        remove_items.append(i_file_name)
                        remove_items.append(o_file_name)
            
                        sftp.get(self.i_list[self.file_index])
                        sftp.get(self.o_list[self.file_index])
            
                        i_file_name = self.extract_file_id(self.i_list[self.file_index])
                        o_file_name = self.extract_file_id(self.o_list[self.file_index])
            
                    remove_items.append(i_file_name)
                    remove_items.append(o_file_name)
                

                    i_features, lab_frame_number = io_function.load_binary_file_frame(i_file_name , self.n_ins)    
                    o_features, out_frame_number = io_function.load_binary_file_frame(o_file_name, self.n_outs)
           
                    for item in remove_items:
                        if os.path.isfile(item) is True:
                            os.remove(item)

                    remove_items = []
                    #############
                    
                
            o_features = o_features[0:frame_number, ]
            i_features = i_features[0:frame_number, ]
            

            if current_index + frame_number <= self.buffer_size:
        
                i_temp_set[current_index:current_index+frame_number, ] = i_features
                o_temp_set[current_index:current_index+frame_number, ] = o_features
                    
                current_index = current_index + frame_number
            else:   ## if current utterance cannot be stored in the block, then leave the remaining part for the next block
                used_frame_number = self.buffer_size - current_index
                i_temp_set[current_index:self.buffer_size, ] = i_features[0:used_frame_number, ]
                o_temp_set[current_index:self.buffer_size, ] = o_features[0:used_frame_number, ]
                current_index = self.buffer_size

                self.i_remain_data = i_features[used_frame_number:frame_number, ]
                self.o_remain_data = o_features[used_frame_number:frame_number, ]
                self.remain_frame_number = frame_number - used_frame_number

            self.file_index += 1
        
        sftp.close()
        
        i_temp_set = i_temp_set[0:current_index, ]
        o_temp_set = o_temp_set[0:current_index, ]

        numpy.random.seed(271639)
        numpy.random.shuffle(i_temp_set)
        numpy.random.seed(271639)
        numpy.random.shuffle(o_temp_set)

        i_shared_set = self.make_shared(i_temp_set, 'x')
        o_shared_set = self.make_shared(o_temp_set, 'y')

        io_shared_set = (i_shared_set, o_shared_set)
        
        return io_shared_set, i_temp_set, o_temp_set
        
        
        
        

# <rawcell>

# FINISH READING

# <codecell>

    def is_finish(self):
        return self.end_reading

# <codecell>

    def extract_file_id(self,file_name):
        file_id = os.path.basename(os.path.splitext(file_name)[0]) + os.path.basename(os.path.splitext(file_name)[1])

        return  file_id

