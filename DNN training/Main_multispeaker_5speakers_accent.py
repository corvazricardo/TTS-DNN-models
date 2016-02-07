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
from C03_Multi_Microsoft import DNN_multi
from C00_Multi import Initial_list


import matplotlib.pyplot as plt


import logging 
import logging.config
import StringIO

# <rawcell>

#     BUILD TRAIN AND TEST LIST DATA

# <codecell>

def make_train_test_list(train_percentage, test_percentage,n_speakers_accent , speaker_accent = 'English'):
        
        try:
            assert train_percentage + test_percentage == 100
        except AssertionError:
            print 'Train and Test values must sum 100%'
            raise
            
        try:
            assert n_speakers_accent > 0
        except AssertionError:
            print "Number of speakers for", speaker_accent , "needs to have one speaker at least"
        

        init_list = Initial_list(n_speakers_accent = n_speakers_accent,speaker_accent = speaker_accent)
        
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

def split_data(io_train_list, io_test_list, train_speaker_ID, test_speaker_ID,  n_ins, n_outs, buffer_size, plot=False):
    
    (i_train_list, o_train_list) = io_train_list
    (i_test_list, o_test_list) = io_test_list
    
    
    n_speakers = len(test_speaker_ID)
    
       
        #Split speakers data
    if n_speakers == 1:
        
        lwr_train = 0
        upr_train = train_speaker_ID[0]
        
        
        lwr_test = 0
        upr_test = test_speaker_ID[0]
        print "Train Data Speaker 1"

        
        train_data_reader_1 = VCTK_feat_collection(i_list=i_train_list[lwr_train:upr_train], o_list= o_train_list[lwr_train:upr_train], n_ins= n_ins, n_outs = n_outs, buffer_size = buffer_size, shuffle = True)
        
        print "Test Data Speaker 1"
        test_data_reader_1 = VCTK_feat_collection(i_list=i_test_list[lwr_test:upr_test], o_list= o_test_list[lwr_test:upr_test], n_ins= n_ins, n_outs = n_outs, buffer_size = buffer_size, shuffle = False)
       

        speakers_data = [(train_data_reader_1, test_data_reader_1)]
        
    elif n_speakers == 2:
            
            
        lwr_train = 0
        upr_train = train_speaker_ID[0]
        
        lwr_test = 0
        upr_test = test_speaker_ID[0]
        print "Train Data Speaker 1"

        
        print "Test Data Speaker 1"

        
        train_data_reader_1 = VCTK_feat_collection(i_list=i_train_list[lwr_train:upr_train], o_list= o_train_list[lwr_train:upr_train], n_ins= n_ins, n_outs = n_outs, buffer_size = buffer_size, shuffle = True)
            
        test_data_reader_1 = VCTK_feat_collection(i_list=i_test_list[lwr_test:upr_test], o_list= o_test_list[lwr_test:upr_test], n_ins= n_ins, n_outs = n_outs, buffer_size = buffer_size, shuffle = False)

        
        lwr_train = upr_train
        upr_train = lwr_train + train_speaker_ID[1]
        
        lwr_test = upr_test
        upr_test = lwr_test + test_speaker_ID[1]
        print "Train Data Speaker 2"

        
        print "Test Data Speaker 2"

        
        
        train_data_reader_2 = VCTK_feat_collection(i_list=i_train_list[lwr_train:upr_train], o_list= o_train_list[lwr_train:upr_train], n_ins= n_ins, n_outs = n_outs, buffer_size = buffer_size, shuffle = True)
            
        test_data_reader_2 = VCTK_feat_collection(i_list=i_test_list[lwr_test:upr_test], o_list= o_test_list[lwr_test:upr_test], n_ins= n_ins, n_outs = n_outs, buffer_size = buffer_size, shuffle = False)

        
        speakers_data = [(train_data_reader_1, test_data_reader_1),(train_data_reader_2, test_data_reader_2)]
            
    elif n_speakers == 3:
        
        lwr_train = 0
        upr_train = train_speaker_ID[0]
        
        lwr_test = 0
        upr_test = test_speaker_ID[0]
        print "Train Data Speaker 1"

        
        print "Test Data Speaker 1"
      
        train_data_reader_1 = VCTK_feat_collection(i_list=i_train_list[lwr_train:upr_train], o_list= o_train_list[lwr_train:upr_train], n_ins= n_ins, n_outs = n_outs, buffer_size = buffer_size, shuffle = True)
            
        test_data_reader_1 = VCTK_feat_collection(i_list=i_test_list[lwr_test:upr_test], o_list= o_test_list[lwr_test:upr_test], n_ins= n_ins, n_outs = n_outs, buffer_size = buffer_size, shuffle = False)

        
        lwr_train = upr_train
        upr_train = lwr_train + train_speaker_ID[1]
        
        lwr_test = upr_test
        upr_test = lwr_test + test_speaker_ID[1]
        print "Train Data Speaker 2"

        
        print "Test Data Speaker 2"

        
        
        train_data_reader_2 = VCTK_feat_collection(i_list=i_train_list[lwr_train:upr_train], o_list= o_train_list[lwr_train:upr_train], n_ins= n_ins, n_outs = n_outs, buffer_size = buffer_size, shuffle = True)
            
        test_data_reader_2 = VCTK_feat_collection(i_list=i_test_list[lwr_test:upr_test], o_list= o_test_list[lwr_test:upr_test], n_ins= n_ins, n_outs = n_outs, buffer_size = buffer_size, shuffle = False)

        lwr_train = upr_train
        upr_train = lwr_train + train_speaker_ID[2]
        
        lwr_test = upr_test
        upr_test = lwr_test + test_speaker_ID[2]
        print "Train Data Speaker 3"

        
        print "Test Data Speaker 3"

        
        
        train_data_reader_3 = VCTK_feat_collection(i_list=i_train_list[lwr_train:upr_train], o_list= o_train_list[lwr_train:upr_train], n_ins= n_ins, n_outs = n_outs, buffer_size = buffer_size, shuffle = True)
        
        test_data_reader_3 = VCTK_feat_collection(i_list=i_test_list[lwr_test:upr_test], o_list= o_test_list[lwr_test:upr_test], n_ins= n_ins, n_outs = n_outs, buffer_size = buffer_size, shuffle = False)

        speakers_data = [(train_data_reader_1, test_data_reader_1),(train_data_reader_2, test_data_reader_2),
                         (train_data_reader_3, test_data_reader_3)]   
    
    elif n_speakers == 4:
    
        
        lwr_train = 0
        upr_train = train_speaker_ID[0]
        
        lwr_test = 0
        upr_test = test_speaker_ID[0]
        print "Train Data Speaker 1"

        
        print "Test Data Speaker 1"

        
        train_data_reader_1 = VCTK_feat_collection(i_list=i_train_list[lwr_train:upr_train], o_list= o_train_list[lwr_train:upr_train], n_ins= n_ins, n_outs = n_outs, buffer_size = buffer_size, shuffle = True)
            
        test_data_reader_1 = VCTK_feat_collection(i_list=i_test_list[lwr_test:upr_test], o_list= o_test_list[lwr_test:upr_test], n_ins= n_ins, n_outs = n_outs, buffer_size = buffer_size, shuffle = False)

        
        lwr_train = upr_train
        upr_train = lwr_train + train_speaker_ID[1]
        
        lwr_test = upr_test
        upr_test = lwr_test + test_speaker_ID[1]
        print "Train Data Speaker 2"
   
        
        print "Test Data Speaker 2"

        
        
        train_data_reader_2 = VCTK_feat_collection(i_list=i_train_list[lwr_train:upr_train], o_list= o_train_list[lwr_train:upr_train], n_ins= n_ins, n_outs = n_outs, buffer_size = buffer_size, shuffle = True)
            
        test_data_reader_2 = VCTK_feat_collection(i_list=i_test_list[lwr_test:upr_test], o_list= o_test_list[lwr_test:upr_test], n_ins= n_ins, n_outs = n_outs, buffer_size = buffer_size, shuffle = False)

        lwr_train = upr_train
        upr_train = lwr_train + train_speaker_ID[2]
        
        lwr_test = upr_test
        upr_test = lwr_test + test_speaker_ID[2]
        print "Train Data Speaker 3"

        
        print "Test Data Speaker 3"

        
        train_data_reader_3 = VCTK_feat_collection(i_list=i_train_list[lwr_train:upr_train], o_list= o_train_list[lwr_train:upr_train], n_ins= n_ins, n_outs = n_outs, buffer_size = buffer_size, shuffle = True)
        
        test_data_reader_3 = VCTK_feat_collection(i_list=i_test_list[lwr_test:upr_test], o_list= o_test_list[lwr_test:upr_test], n_ins= n_ins, n_outs = n_outs, buffer_size = buffer_size, shuffle = False)

        lwr_train = upr_train
        upr_train = lwr_train + train_speaker_ID[3]
        
        lwr_test = upr_test
        upr_test = lwr_test + test_speaker_ID[3]
        print "Train Data Speaker 4"

        
        print "Test Data Speaker 4"

        
        train_data_reader_4 = VCTK_feat_collection(i_list=i_train_list[lwr_train:upr_train], o_list= o_train_list[lwr_train:upr_train], n_ins= n_ins, n_outs = n_outs, buffer_size = buffer_size, shuffle = True)
        
        test_data_reader_4 = VCTK_feat_collection(i_list=i_test_list[lwr_test:upr_test], o_list= o_test_list[lwr_test:upr_test], n_ins= n_ins, n_outs = n_outs, buffer_size = buffer_size, shuffle = False)

        
        speakers_data = [(train_data_reader_1, test_data_reader_1),(train_data_reader_2, test_data_reader_2),
                         (train_data_reader_3, test_data_reader_3),(train_data_reader_4, test_data_reader_4)] 
        
            
    elif n_speakers == 5:
            
        lwr_train = 0
        upr_train = train_speaker_ID[0]
        
        lwr_test = 0
        upr_test = test_speaker_ID[0]
        
        print "Train Data Speaker 1"

        
        print "Test Data Speaker 1"

        
        train_data_reader_1 = VCTK_feat_collection(i_list=i_train_list[lwr_train:upr_train], o_list= o_train_list[lwr_train:upr_train], n_ins= n_ins, n_outs = n_outs, buffer_size = buffer_size, shuffle = True)
            
        test_data_reader_1 = VCTK_feat_collection(i_list=i_test_list[lwr_test:upr_test], o_list= o_test_list[lwr_test:upr_test], n_ins= n_ins, n_outs = n_outs, buffer_size = buffer_size, shuffle = False)

        
        lwr_train = upr_train
        upr_train = lwr_train + train_speaker_ID[1]
        
        lwr_test = upr_test
        upr_test = lwr_test + test_speaker_ID[1]
        print "Train Data Speaker 2"
  
        
        print "Test Data Speaker 2"
    
        
        
        train_data_reader_2 = VCTK_feat_collection(i_list=i_train_list[lwr_train:upr_train], o_list= o_train_list[lwr_train:upr_train], n_ins= n_ins, n_outs = n_outs, buffer_size = buffer_size, shuffle = True)
            
        test_data_reader_2 = VCTK_feat_collection(i_list=i_test_list[lwr_test:upr_test], o_list= o_test_list[lwr_test:upr_test], n_ins= n_ins, n_outs = n_outs, buffer_size = buffer_size, shuffle = False)

        lwr_train = upr_train
        upr_train = lwr_train + train_speaker_ID[2]
        
        lwr_test = upr_test
        upr_test = lwr_test + test_speaker_ID[2]
        print "Train Data Speaker 3"

        
        print "Test Data Speaker 3"

        
        train_data_reader_3 = VCTK_feat_collection(i_list=i_train_list[lwr_train:upr_train], o_list= o_train_list[lwr_train:upr_train], n_ins= n_ins, n_outs = n_outs, buffer_size = buffer_size, shuffle = True)
        
        test_data_reader_3 = VCTK_feat_collection(i_list=i_test_list[lwr_test:upr_test], o_list= o_test_list[lwr_test:upr_test], n_ins= n_ins, n_outs = n_outs, buffer_size = buffer_size, shuffle = False)

        lwr_train = upr_train
        upr_train = lwr_train + train_speaker_ID[3]
        
        lwr_test = upr_test
        upr_test = lwr_test + test_speaker_ID[3]
        print "Train Data Speaker 4"

        
        print "Test Data Speaker 4"

        
        train_data_reader_4 = VCTK_feat_collection(i_list=i_train_list[lwr_train:upr_train], o_list= o_train_list[lwr_train:upr_train], n_ins= n_ins, n_outs = n_outs, buffer_size = buffer_size, shuffle = True)
        
        test_data_reader_4 = VCTK_feat_collection(i_list=i_test_list[lwr_test:upr_test], o_list= o_test_list[lwr_test:upr_test], n_ins= n_ins, n_outs = n_outs, buffer_size = buffer_size, shuffle = False)

        lwr_train = upr_train
        upr_train = lwr_train + train_speaker_ID[4]
        
        lwr_test = upr_test
        upr_test = lwr_test + test_speaker_ID[4]
        print "Train Data Speaker 5"

        
        print "Test Data Speaker 5"

        
        
        train_data_reader_5 = VCTK_feat_collection(i_list=i_train_list[lwr_train:upr_train], o_list= o_train_list[lwr_train:upr_train], n_ins= n_ins, n_outs = n_outs, buffer_size = buffer_size, shuffle = True)
        
        test_data_reader_5 = VCTK_feat_collection(i_list=i_test_list[lwr_test:upr_test], o_list= o_test_list[lwr_test:upr_test], n_ins= n_ins, n_outs = n_outs, buffer_size = buffer_size, shuffle = False)

        
        speakers_data = [(train_data_reader_1, test_data_reader_1),(train_data_reader_2, test_data_reader_2),
                         (train_data_reader_3, test_data_reader_3),(train_data_reader_4, test_data_reader_4),
                         (train_data_reader_5, test_data_reader_5)] 
    
            
        
    return speakers_data
    
    

# <codecell>

def train_of_DNN(all_speakers_data,all_dnn_models,all_train_fns,all_valid_fns,buffer_size):
    
    speakers_data_1,speakers_data_2 = all_speakers_data['1'],all_speakers_data['2']
    dnn_models_1,dnn_models_2 = all_dnn_models['1'],all_dnn_models['2']
    train_fns_1,train_fns_2 = all_train_fns['1'],all_train_fns['2']
    valid_fns_1,valid_fns_2 = all_valid_fns['1'],all_valid_fns['2']
    
    total_speakers = len(speakers_data_1) + len(speakers_data_2)
    
    buffer_size = buffer_size
        

    learning_rate = 0.0002
    finetune_lr = learning_rate
    training_epochs=100
        

    batch_size = 256#973
    l1_reg = 0.0 #L1_regularization
    l2_reg = 0.00001 #L2_regularization
    private_l2_reg = 0.00001
    warmup_epoch = 5
        
    #warmup_epoch = 10
    momentum = 0.9
    warmup_momentum = 0.3
      
    #Check function train_of_DNN to have the same value in the hidden layers sizes    
    hidden_layers_sizes = [512,512,512,512,512,512]
        
    stream_weights = [1.0]
    private_hidden_sizes = [1024]
        
    buffer_utt_size = 400
    early_stop_epoch = 35
    
    
    start_time = time.clock()
       

    early_stop = 0
    epoch = 0
    global_early_stop = 0
    previous_finetune_lr = finetune_lr
    values = []
    
    new_param_list = [0 for i in range(0,total_speakers)]
                                       
    old_param_list = [0 for i in range(0,total_speakers)]
    
    new_dparam_list = [0 for i in range(0,total_speakers)]
    
    old_dparam_list = [0 for i in range(0,total_speakers)]
    plot = []
    
    while (epoch < training_epochs):
        epoch = epoch + 1
        print "-----------------------Epoch  :", epoch 
        if epoch == 1:
            best_validation_loss = sys.float_info.max
            previous_loss = sys.float_info.max
        
        current_momentum = momentum
        current_finetune_lr = finetune_lr
        
        train_error = []
        sub_start_time = time.clock()
        
        
        
        for model in range(0,total_speakers):
            
            print "DNN model :", model, "out of", total_speakers-1, "models"
            print "Epoch :", epoch
            
            #Setting models of the first accent
            if model < len(speakers_data_1):
                
                dnn_model = dnn_models_1[str(model)]
                
                train_fn = train_fns_1[str(model)]
                valid_fn = valid_fns_1[str(model)]
                
                train_data = speakers_data_1[model][0]
                test_data = speakers_data_1[model][1]
                
                if epoch ==1:
                    
                    best_dnn_model = dnn_model
                    
                    new_param_list[model] = dnn_model.params
                    new_dparam_list[model] = dnn_model.delta_params 
                    
                    old_param_list[model] = dnn_model.params
                    old_dparam_list[model] = dnn_model.delta_params 
                    
                    
                    dnn_model.best_validation_loss = sys.float_info.max
                    dnn_model.previous_loss = sys.float_info.max
                    
                else:
                    
                    
                    old_param_list[model] = dnn_model.params
                    old_dparam_list[model] = dnn_model.delta_params
                    
                    

            
            #Setting models of the second accent
            else:
                
                dnn_model = dnn_models_2[str(model-len(speakers_data_1))]
                train_fn = train_fns_2[str(model-len(speakers_data_1))]
                valid_fn = valid_fns_2[str(model-len(speakers_data_1))]
                
                train_data = speakers_data_2[model-len(speakers_data_1)][0]
                test_data = speakers_data_2[model-len(speakers_data_1)][1]
                
                if epoch ==1:
                    
                    best_dnn_model = dnn_model
                    
                    new_param_list[model] = dnn_model.params
                    
                    dnn_model.best_validation_loss = sys.float_info.max
                    dnn_model.previous_loss = sys.float_info.max
                    
                else: 
                    
                    old_param_list[model] = dnn_model.params
                    old_dparam_list[model] = dnn_model.delta_params

            
            current_momentum = momentum
            current_finetune_lr = finetune_lr
            
            dnn_model.train_error = []
            
            if not train_data.is_finish():
                
                print "Data of speaker",model, "loaded"
                io_shared_train_set, i_temp_train_set, o_temp_train_set = train_data.load_next_partition()
                i_train_set, o_train_set = io_shared_train_set
                
                i_train_set.set_value(numpy.asarray(i_temp_train_set, dtype=theano.config.floatX), borrow=True)
                o_train_set.set_value(numpy.asarray(o_temp_train_set, dtype=theano.config.floatX), borrow=True)
                
                n_train_batches = i_train_set.get_value().shape[0] / batch_size
                
                print " This partition :",i_train_set.get_value(borrow=True).shape[0], "frames (divided into ", n_train_batches, "of size", batch_size
          
  
                
                
                for minibatch_index in xrange(n_train_batches):
                        this_train_error = train_fn(minibatch_index, current_finetune_lr, current_momentum)
                        dnn_model.train_error.append(this_train_error)
                
                print 'Calculating validation loss'
                validation_losses = valid_fn()
                this_validation_loss = numpy.mean(validation_losses)
                
                # this has a possible bias if the minibatches were not all of identical size
                # but it should not be siginficant if minibatches are small
                
                #Saving train Error
                dnn_model.this_train_valid_loss = numpy.mean(dnn_model.train_error)
                
                loss_difference = this_validation_loss - dnn_model.previous_loss
                
                print "Epoch :", epoch, "Speaker :",model," Validation Error :", this_validation_loss, " Train Error :", dnn_model.this_train_valid_loss

                
                
                values.append((epoch,model,this_validation_loss))
                

                
                
                old_param_list[model] = best_dnn_model.params
                old_dparam_list[model] = best_dnn_model.delta_params
                

                    
                best_dnn_model = dnn_model
                best_dnn_model.previous_loss = this_validation_loss
                new_param_list[model] = best_dnn_model.params
                new_dparam_list[model] = best_dnn_model.delta_params
                    
                    
                   
                    
                if model < len(speakers_data_1):
                    dnn_models_1[str(model)] = best_dnn_model
                    dnn_models_1[str(model)].best_validation_loss = this_validation_loss
                else:
                    dnn_models_2[str(model-len(speakers_data_1))] = best_dnn_model
                    dnn_models_2[str(model-len(speakers_data_1))].best_validation_loss = this_validation_loss
                        
                print "Validation loss decreased, so saving model"
                
                
                    

        early_stop += 1
                    
            
         
    
        for i in range(0,total_speakers):
            

            if epoch != 1:
                
                
    
                if old_param_list[i][0:-2] != new_param_list[i][0-2] :
                
                    for speaker in range(0,total_speakers):
                    
                        if speaker < len(speakers_data_1):
                            '''
                            print "Data from accent 1-1 if"
                            print "Speaker", speaker
                            print "Model",i
                            '''
                            size_shared = len(new_param_list[i][:-2])
                            size_accent = len(new_param_list[i][:-4])
                            
                            for state in range(0,size_shared,2):
                                
                                W_old = dnn_models_1[str(speaker)].params[state].get_value()
                                W_new = new_param_list[i][state].get_value()
                                delta = W_old - W_new
                                W = W_old - delta
          
                                dnn_models_1[str(speaker)].params[state].set_value(W)
                            
                            for state in range(0,size_accent,2):
                                
                                W_old = dnn_models_2[str(speaker)].params[state].get_value()
                                W_new = new_param_list[i][state].get_value()
                                delta = W_old - W_new
                                W = W_old - delta
                    
                                dnn_models_2[str(speaker)].params[state].set_value(W)

            
            
            
                        
                        else:
 
                            
                            size_shared = len(new_param_list[i][:-2])
                            size_accent = len(new_param_list[i][:-4])
                            
                            for state in range(0,size_accent,2):
                                
                                W_old = dnn_models_1[str(speaker-len(speakers_data_1))].params[state].get_value()
                                W_new = new_param_list[i][state].get_value()
                                delta = W_old - W_new
                                W = W_old - delta

                                dnn_models_1[str(speaker-len(speakers_data_1))].params[state].set_value(W)
                            
                            for state in range(0,size_shared,2):
                                
                                W_old = dnn_models_2[str(speaker-len(speakers_data_1))].params[state].get_value()
                                W_new = new_param_list[i][state].get_value()
                                delta = W_old - W_new
                                W = W_old - delta
 
                                dnn_models_2[str(speaker-len(speakers_data_1))].params[state].set_value(W)
                        

                
            else:
                
                for speaker in range(0,total_speakers):
                    
                        if speaker < len(speakers_data_1):

                            
                            size_shared = len(new_param_list[i][:-2])
                            size_accent = len(new_param_list[i][:-4])
                            
                            for state in range(0,size_shared,2):
                                
                                W_old = dnn_models_1[str(speaker)].params[state].get_value()
                                W_new = new_param_list[i][state].get_value()
                                delta = W_old - W_new
                                W = W_old - delta

                                dnn_models_1[str(speaker)].params[state].set_value(W)
                            
                            for state in range(0,size_accent,2):
                                
                                W_old = dnn_models_2[str(speaker)].params[state].get_value()
                                W_new = new_param_list[i][state].get_value()
                                delta = W_old - W_new
                                W = W_old - delta
   
                                dnn_models_2[str(speaker)].params[state].set_value(W)
                            
                            
                            
         
                        
                        else:

                            
                            size_shared = len(new_param_list[i][:-2])
                            size_accent = len(new_param_list[i][:-4])
                            
                            for state in range(0,size_accent,2):
                                
                                W_old = dnn_models_1[str(speaker-len(speakers_data_1))].params[state].get_value()
                                W_new = new_param_list[i][state].get_value()
                                delta = W_old - W_new
                                W = W_old - delta
                  
                                dnn_models_1[str(speaker-len(speakers_data_1))].params[state].set_value(W)
                            
                            for state in range(0,size_shared,2):
                                
                                W_old = dnn_models_2[str(speaker-len(speakers_data_1))].params[state].get_value()
                                W_new = new_param_list[i][state].get_value()
                                delta = W_old - W_new
                                W = W_old - delta
                   
                                dnn_models_2[str(speaker-len(speakers_data_1))].params[state].set_value(W)

                            
                     

            
            
                
                
        if early_stop > early_stop_epoch:
                print "Stopping early"
                
                return values 
                
                break        
                  
                
                

# <codecell>

def get_models_accent( io_train_list, io_test_list, train_speaker_ID, test_speaker_ID,  n_ins, n_outs, buffer_size, plot=False):

        print "Starting train_DNN of Microsoft Paper"
        

        
        
        ##################Setting parameters#### 
        dnn_models = {}
        train_fns = {}
        valid_fns = {}
        
        batch_size = 256#973
        l1_reg = 0.0 #L1_regularization
        l2_reg = 0.00001 #L2_regularization
        
        
        #Check function train_of_DNN to have the same value in the hidden layers sizes
        hidden_layers_sizes = [512,512,512,512,512,512]
        
        hidden_activation = 'tanh'
        output_activation = 'linear'

        buffer_size = int(buffer_size / batch_size) * batch_size
        

        speakers_data = split_data(io_train_list, io_test_list, train_speaker_ID, test_speaker_ID,  n_ins=n_ins, n_outs=n_outs, buffer_size=buffer_size, plot=False)

        model_data_reader = [() for i in speakers_data]
        
        # numpy random generator
        numpy_rng = numpy.random.RandomState(123)
        
        for i in range(0,len(speakers_data)):
            
            dnn_model = None
            train_fn = None
            valid_fn = None
            valid_model = None
            
             
            io_shared_train_set, i_temp_train_set, o_temp_train_set = speakers_data[i][0].load_next_partition()
            i_train_set, o_train_set = io_shared_train_set
        
            io_shared_test_set, i_temp_test_set, o_temp_test_set = speakers_data[i][1].load_next_partition()
            i_test_set, o_test_set = io_shared_test_set
            
            speakers_data[i][0].reset()
            speakers_data[i][1].reset()
            
           
            print "Buiding model for speaker  :", i+1
            
            
            dnn_model = DNN_multi(numpy_rng=numpy_rng, n_ins=n_ins, n_outs = n_outs,
                        l1_reg = l1_reg, l2_reg = l2_reg, 
                         hidden_layers_sizes = hidden_layers_sizes,
                         hidden_activation = hidden_activation, 
                         output_activation = output_activation)
            
            train_fn, valid_fn = dnn_model.build_finetune_functions(
                    (i_train_set, o_train_set), (i_test_set, o_test_set), batch_size=batch_size)
            
            if i == 0:
                dnn_models['0'] = dnn_model
                train_fns['0'] = train_fn
                valid_fns['0'] = valid_fn

            elif i == 1:
                dnn_models['1'] = dnn_model
                train_fns['1'] = train_fn
                valid_fns['1'] = valid_fn
            elif i == 2:
                dnn_models['2'] = dnn_model
                train_fns['2'] = train_fn
                valid_fns['2'] = valid_fn
            elif i == 3:
                dnn_models['3'] = dnn_model
                train_fns['3'] = train_fn
                valid_fns['3'] = valid_fn
            elif i == 4:
                dnn_models['4'] = dnn_model
                train_fns['4'] = train_fn
                valid_fns['4'] = valid_fn
                
 
        
        return (speakers_data,dnn_models,train_fns,valid_fns)
        
        

# <codecell>



io_train_list_1, io_test_list_1, train_speaker_ID_1, test_speaker_ID_1 = make_train_test_list(80,20,n_speakers_accent= 5, speaker_accent = 'English')

io_train_list_2, io_test_list_2, train_speaker_ID_2, test_speaker_ID_2 = make_train_test_list(80,20,n_speakers_accent= 5, speaker_accent = 'Scottish')




# <codecell>

buffer_size = 5000


# <codecell>

speakers_data_1,dnn_models_1,train_fns_1,valid_fns_1 = get_models_accent(io_train_list_1,io_test_list_1, train_speaker_ID_1, test_speaker_ID_1, n_ins=601,n_outs =259, buffer_size = buffer_size, plot = False)

speakers_data_2,dnn_models_2,train_fns_2,valid_fns_2 = get_models_accent(io_train_list_2,io_test_list_2, train_speaker_ID_2, test_speaker_ID_2, n_ins=601,n_outs =259, buffer_size = buffer_size, plot = False)

all_speakers_data = {'1':speakers_data_1,'2':speakers_data_2}
all_dnn_models = {'1':dnn_models_1,'2':dnn_models_2}
all_train_fns = {'1':train_fns_1,'2':train_fns_2}
all_valid_fns = {'1':valid_fns_1,'2':valid_fns_2}

# <codecell>


values = train_of_DNN(all_speakers_data,all_dnn_models,all_train_fns,all_valid_fns,buffer_size)

# <codecell>


# <codecell>

vector_1 = []
vector_2 = []
vector_3 = []
vector_4 = []
vector_5 = []
vector_6 = []
vector_7 = []
vector_8 = []
vector_9 = []
vector_10 = []


epoch = []
epochs =50
for i in values:
    if i[1]==0 and i[1]<=epochs:
        vector_1.append(i[2])
        epoch.append(i[0])
    elif i[1]==1 and i[1]<=epochs :
        if i[0]<=epochs:
            vector_2.append(i[2])
    elif i[1]==2 and i[1]<=epochs :
        if i[0]<=epochs:
            vector_3.append(i[2])
    
    elif i[1]==3 and i[1]<=epochs :
        if i[0]<=epochs:
            vector_4.append(i[2])
    elif i[1]==4 and i[1]<=epochs :
        if i[0]<=epochs:
            vector_5.append(i[2])
    elif i[1]==5 and i[1]<=epochs :
        if i[0]<=epochs:
            vector_6.append(i[2])
    elif i[1]==6 and i[1]<=epochs :
        if i[0]<=epochs:
            vector_7.append(i[2])
    elif i[1]==7 and i[1]<=epochs :
        if i[0]<=epochs:
            vector_8.append(i[2])
    elif i[1]==8 and i[1]<=epochs :
        if i[0]<=epochs:
            vector_9.append(i[2])
    elif i[1]==9 and i[1]<=epochs :
        if i[0]<=epochs:
            vector_10.append(i[2])

# <codecell>

from matplotlib.legend_handler import HandlerLine2D
plt.figure(1)
plt.subplot(211)
plt.title('Multi-speaker model with 5 English accent speakers',fontsize=14)
plt.xlabel('Epoch')
plt.ylabel('RMSE')
plt.grid(True)
speaker1, = plt.plot(vector_1,label = 'Speaker 1',linestyle = '-',marker='^',markersize=7,linewidth=2.0)
speaker2, = plt.plot(vector_2,label = 'Speaker 2',linestyle = '-',marker='o',markersize=7,linewidth=2.0)
speaker3, = plt.plot(vector_3,label = 'Speaker 3',linestyle = '-',marker='o',markersize=7,linewidth=2.0)
speaker4, = plt.plot(vector_4,label = 'Speaker 4',linestyle = '-',marker='o',markersize=7,linewidth=2.0)
speaker5, = plt.plot(vector_5,label = 'Speaker 5',linestyle = '-',marker='o',markersize=7,linewidth=2.0)

plt.legend(handler_map={speaker1: HandlerLine2D(numpoints=1)},prop={'size':9},loc='center right')



plt.subplot(212)
plt.title('Multi-speaker model with 5 Scottish accent speakers',fontsize=14)
plt.xlabel('Epoch')
plt.ylabel('RMSE')
plt.grid(True)
speaker1, = plt.plot(vector_6,label = 'Speaker 1',linestyle = '-',marker='^',markersize=7,linewidth=2.0)
speaker2, = plt.plot(vector_7,label = 'Speaker 2',linestyle = '-',marker='o',markersize=7,linewidth=2.0)
speaker3, = plt.plot(vector_8,label = 'Speaker 3',linestyle = '-',marker='o',markersize=7,linewidth=2.0)
speaker4, = plt.plot(vector_9,label = 'Speaker 4',linestyle = '-',marker='o',markersize=7,linewidth=2.0)
speaker5, = plt.plot(vector_10,label = 'Speaker 5',linestyle = '-',marker='o',markersize=7,linewidth=2.0)

plt.legend(handler_map={speaker1: HandlerLine2D(numpoints=1)},prop={'size':9},loc='bottom right')
plt.show()

# <codecell>


