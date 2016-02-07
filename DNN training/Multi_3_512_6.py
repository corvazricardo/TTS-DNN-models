# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

# Part of the code was used from code of Dr. Zhizheng Wu.
# Code edited and extended by R. Cortez
# Author: Dr. Zhizheng Wu and R. Cortez
# The University of Edinburgh
# Version: Aug 19 2015

import cPickle
import gzip
import os, sys, errno
import time
import math

#  numpy & theano imports need to be done in this order (only for some numpy installations, not sure why)
import numpy
# we need to explicitly import this in some cases, not sure why this doesn't get imported with numpy itself
import numpy.distutils.__config__
# and only after that can we import theano 
import theano

from C01 import VCTK_feat_collection
from C02 import DNN
from C00 import Initial_list


import matplotlib.pyplot as plt

import logging 
import logging.config
import StringIO

# <rawcell>

#     BUILD TRAIN AND TEST LIST DATA

# <codecell>

def make_train_test_list(train_percentage, test_percentage, n_speakers, speaker_accent):
        
        try:
            assert train_percentage + test_percentage == 100
        except AssertionError:
            print 'Train and Test values must sum 100%'
            raise
                    
        init_list = Initial_list(n_speakers = n_speakers,speaker_accent = speaker_accent)
        train_speaker_ID = []
        test_speaker_ID = []
        i_train_list = []
        o_train_list = []
        i_test_list = []
        o_test_list = []
        
        
        io_train_list = []
        io_test_list = []
        
        i_list = init_list.i_list
        o_list = init_list.o_list
        speakers = init_list.speaker_ID
        speakers_data = len(speakers)
        index = 0
        
        for partition in speakers: 
            train_value = int((partition)* (train_percentage/100.))
            train = 0
            test = 0
            for file_name in range(index,partition+index): 
                
                if file_name <= (train_value + index):
                    i_train_list.append(i_list[file_name])
                    o_train_list.append(o_list[file_name])
                    train = train + 1
                    
                elif file_name > (train_value + index):
                    i_test_list.append(i_list[file_name])
                    o_test_list.append(o_list[file_name])
                    test = test + 1
                    
                
            train_speaker_ID.append(train)
            test_speaker_ID.append(test)
            index = index + partition
        
          
        
        io_train_list = (i_train_list, o_train_list)
        io_test_list = (i_test_list, o_test_list)
        
        return io_train_list,io_test_list, train_speaker_ID, test_speaker_ID
        
       

# <rawcell>

# BUILD FILE ID LIST

# <codecell>

def extract_file_id_list(file_list):
    file_id_list = []
    for file_name in file_list:
        file_id = os.path.basename(os.path.splitext(file_name)[0])
        file_id_list.append(file_id)

    return  file_id_list

# <rawcell>

# READ FILE LIST "PROBABLY WE DON'T NEED THIS METHOD"

# <codecell>

def read_file_list(file_name):

    file_lists = []
    fid = open(file_name)
    for line in fid.readlines():
        line = line.strip()
        if len(line) < 1:
            continue
        file_lists.append(line)
    fid.close()
    
    print "read_file_list from :", file_name
    
    return  file_lists

# <rawcell>

# MAKE OUTPUT FILE LIST   "PROBABLY WE DON'T NEED THIS METHOD"

# <codecell>

def make_output_file_list(out_dir, in_file_lists):
    out_file_lists = []

    for in_file_name in in_file_lists:
        file_id = os.path.basename(in_file_name)
        out_file_name = out_dir + '/' + file_id
        out_file_lists.append(out_file_name)

    return  out_file_lists

# <rawcell>

# PREPARE FILE PATH LIST "PROBABLY WE DON'T NEED THIS METHOD"

# <codecell>

def prepare_file_path_list(file_id_list, file_dir, file_extension, new_dir_switch=True):
    if not os.path.exists(file_dir) and new_dir_switch:
        os.makedirs(file_dir)
    file_name_list = []
    for file_id in file_id_list:
        file_name = file_dir + '/' + file_id + file_extension
        file_name_list.append(file_name)
        base_dir = os.path.dirname(os.path.abspath(file_name))
        if not os.path.exists(base_dir) and new_dir_switch:
            os.makedirs(base_dir)

    return  file_name_list

# <rawcell>

# VISUALIZE DNN

# <codecell>

def visualize_dnn(dnn):
    
    layer_num = len(dnn.params) / 2     ## including input and output
    
    for i in xrange(layer_num):
        fig_name = 'Activation weights W' + str(i)
        fig_title = 'Activation weights of W' + str(i)
        xlabel = 'Neuron index of hidden layer ' + str(i)
        ylabel = 'Neuron index of hidden layer ' + str(i+1)
        if i == 0:
            xlabel = 'Input feature index'
        if i == layer_num-1:
            ylabel = 'Output feature index'
        
        logger.create_plot(fig_name, SingleWeightMatrixPlot)
        plotlogger.add_plot_point(fig_name, fig_name, dnn.params[i*2].get_value(borrow=True).T)
        plotlogger.save_plot(fig_name, title=fig_name, xlabel=xlabel, ylabel=ylabel)

# <rawcell>

# TRAIN DNN

# <codecell>

def train_DNN( io_train_list, io_test_list, n_ins, n_outs, n_speakers, buffer_size, plot=False):
        #################consider what is nnest_file_name about
        print "Starting train_DNN"
        

        
        ##################Setting parameters#### 
        #n_ins = 601
        #n_outs = 259
        n_speakers = n_speakers

        #buffer_size = 200000
        buffer_size = buffer_size
        
        #learning_rate = 0.012
        learning_rate = 0.0002
        finetune_lr = learning_rate
        training_epochs=100
        

        batch_size = 256
        l1_reg = 0.0 #L1_regularization
        l2_reg = 0.00001 #L2_regularization
        private_l2_reg = 0.00001
        warmup_epoch = 5
        #warmup_epoch = 10
        momentum = 0.9
        warmup_momentum = 0.3
	hidden_layers_sizes = [512,512,512,512,512,512]



        stream_weights = [1.0]
        private_hidden_sizes = [1024]
        
        buffer_utt_size = 400
        early_stop_epoch = 5
        
        hidden_activation = 'tanh'
        output_activation = 'linear'
        #stream_lr_weights
        #use_private_hidden
        model_type = 'DNN'

        #self.speaker_ID= speaker_ID
        #print "Speaker ID :", self.speaker_ID
        
        
        ## use a switch to turn on pretraining
        ## pretraining may not help too much, if this case, we turn it off to save time
        do_pretraining = False
        pretraining_epochs = 10
        pretraining_lr = 0.0001


        buffer_size = int(buffer_size / batch_size) * batch_size
        
        ###################

        (i_train_list, o_train_list) = io_train_list
        (i_test_list, o_test_list) = io_test_list
        
        print "Building training data provider"
        
        train_data_reader = VCTK_feat_collection(i_list=i_train_list, o_list= o_train_list, n_ins= n_ins, n_outs = n_outs, n_speakers= n_speakers, buffer_size = buffer_size, shuffle = True)
        
        print "Building testing data provider"
            
        test_data_reader = VCTK_feat_collection(i_list=i_test_list, o_list= o_test_list, n_ins= n_ins, n_outs = n_outs, n_speakers= n_speakers, buffer_size = buffer_size, shuffle = False)
        
        
        io_shared_train_set, i_temp_train_set, o_temp_train_set = train_data_reader.load_next_partition()
        i_train_set, o_train_set = io_shared_train_set
        
        io_shared_test_set, i_temp_test_set, o_temp_test_set = test_data_reader.load_next_partition()
        i_test_set, o_test_set = io_shared_test_set
        train_data_reader.reset()
        test_data_reader.reset()
        
        ##temporally we use the training set as pretrain_set_x.
        ##we need to support any data for pretraining
        i_pretrain_set = i_train_set

        # numpy random generator
        numpy_rng = numpy.random.RandomState(123)
        print "Buiding the model"
        
        dnn_model = None
        pretrain_fn = None  ## not all the model support pretraining right now
        train_fn = None
        valid_fn = None
        valid_model = None ## valid_fn and valid_model are the same. reserve to computer multi-stream distortion
        
        if model_type == 'DNN':
            dnn_model = DNN(numpy_rng=numpy_rng, n_ins=n_ins, n_outs = n_outs,
                        l1_reg = l1_reg, l2_reg = l2_reg, 
                         hidden_layers_sizes = hidden_layers_sizes, 
                          hidden_activation = hidden_activation, 
                          output_activation = output_activation)
            train_fn, valid_fn = dnn_model.build_finetune_functions(
                    (i_train_set, o_train_set), (i_test_set, o_test_set), batch_size=batch_size)

 
        
        print "Fine-tuning the ", model_type, "model"
        
        start_time = time.clock()

        best_dnn_model = dnn_model
        best_validation_loss = sys.float_info.max
        previous_loss = sys.float_info.max

        early_stop = 0
        epoch = 0
        previous_finetune_lr = finetune_lr
        
        while (epoch < training_epochs):
            epoch = epoch + 1
        
            current_momentum = momentum
            current_finetune_lr = finetune_lr
            if epoch <= warmup_epoch:
                current_finetune_lr = finetune_lr
                current_momentum = warmup_momentum
            else:
                current_finetune_lr = previous_finetune_lr * 0.5
        
            previous_finetune_lr = current_finetune_lr
        
            train_error = []
            sub_start_time = time.clock()
            
            while (not train_data_reader.is_finish()):
                io_shared_train_set, i_temp_train_set, o_temp_train_set = train_data_reader.load_next_partition()
                i_train_set.set_value(numpy.asarray(i_temp_train_set, dtype=theano.config.floatX), borrow=True)
                o_train_set.set_value(numpy.asarray(o_temp_train_set, dtype=theano.config.floatX), borrow=True)
            
                n_train_batches = i_train_set.get_value().shape[0] / batch_size
                
                print " This partition :",i_train_set.get_value(borrow=True).shape[0], "frames (divided into ", n_train_batches, "of size", batch_size
          
                for minibatch_index in xrange(n_train_batches):
                    this_train_error = train_fn(minibatch_index, current_finetune_lr, current_momentum)
                    train_error.append(this_train_error)
                
                    if numpy.isnan(this_train_error):
                        print "Training error over minibatch ", minibatch_index+1, " of ", n_train_batches, " was ", this_train_error
           
            train_data_reader.reset()
            
            print 'Calculating validation loss'
            validation_losses = valid_fn()
            this_validation_loss = numpy.mean(validation_losses)
            
            # this has a possible bias if the minibatches were not all of identical size
            # but it should not be siginficant if minibatches are small
            this_train_valid_loss = numpy.mean(train_error)

            sub_end_time = time.clock()

            loss_difference = this_validation_loss - previous_loss

            print "Epoch :", epoch, " Validation Error :", this_validation_loss, " Train Error :", this_train_valid_loss, " Time Spent : ",(sub_end_time - sub_start_time)
            

            
            if this_validation_loss < best_validation_loss:
                best_dnn_model = dnn_model
                best_validation_loss = this_validation_loss
                print "Validation loss decreased, so saving model"
                
            if this_validation_loss >= previous_loss:
                print "Validation loss increased"
                dbn = best_dnn_model
                early_stop += 1
            
            if early_stop > early_stop_epoch:
                print "Stopping early"
                break
            
            if math.isnan(this_validation_loss):
                break
                
            previous_loss = this_validation_loss
            
        end_time = time.clock()
    
        #####OJO CON ESTA LINEA#####################################
        cPickle.dump(best_dnn_model, open("output.data", 'wb'))  
    
        print "Overall training time : ", ((end_time - start_time) / 60.), " Validation Error :", best_validation_loss
        print "Architecture Training :",hidden_layers_sizes
        print "Number of speakers :", n_speakers, 
        return best_validation_loss
         

# <rawcell>

# DNN GENERATION

# <codecell>   
    
    

# <codecell>

io_train_list, io_test_list, train_speaker_ID, test_speaker_ID = make_train_test_list(80,20,3,speaker_accent = 'Scottish')

# <codecell>

best_validation_loss = train_DNN(io_train_list,io_test_list,n_ins=601,n_outs =259,n_speakers = 3, buffer_size = 5000, plot = False)


# <codecell>


