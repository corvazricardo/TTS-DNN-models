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
import theano.tensor as T

from scipy import linalg 


from C01_Multi import VCTK_feat_collection
import matplotlib.pyplot as plt
import StringIO

from layers import LinearLayer, HiddenLayer, SigmoidLayer


from theano.tensor.shared_randomstreams import RandomStreams

theano.config.floatX = 'float32'

# <rawcell>

# CREATE DNN OBJECT

# <codecell>

class DNN_multi(object):
    
    def __init__(self, numpy_rng = numpy.random.RandomState(2**30), theano_rng=None, n_ins=601,
                 n_outs=259, l1_reg = None, l2_reg = None, 
                 hidden_layers_sizes= [512,512,512,512,512,512,512],
                 n_speakers_accent = 2,
                 hidden_activation='tanh', output_activation='linear'):
        
        print "DNN MULTI-SPEAKER INITIALISATION"
        
        self.sigmoid_layers = []
        self.params = []
        self.delta_params = []
        self.n_layers = len(hidden_layers_sizes)
        

        self.n_ins = n_ins
        self.n_outs = n_outs
        
        self.output_activation = output_activation
        
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
        
        
        self.final_layer_accent = []
        self.error_cost = []
        
        #finetune_cost = []
        #self.finetune_costs_accent = []
        
        self.errors_accent = []
             
        
        assert self.n_layers > 0
        
        if not theano_rng:
            theano_rng = RandomStreams(numpy.random.randint(2 ** 30))

        # allocate symbolic variables for the data
        self.x = T.matrix('x') 
        self.y = T.matrix('y')
        
        for i in xrange(self.n_layers):
            
            if i==0:
                
                input_size = n_ins
            else:
                
                input_size = hidden_layers_sizes [i-1]
            
            if i==0:
                
                layer_input = self.x
            else:
                 
                layer_input = self.sigmoid_layers[-1].output
            
            
            sigmoid_layer = HiddenLayer(rng=numpy_rng,
                                        input=layer_input,
                                        n_in = input_size,
                                        n_out = hidden_layers_sizes[i],
                                        activation=T.tanh)
            

            self.sigmoid_layers.append(sigmoid_layer)
            self.params.extend(sigmoid_layer.params)
            self.delta_params.extend(sigmoid_layer.delta_params)
         
         
        ####Final Layer for speaker
        
            
        if self.output_activation == 'linear':
            self.final_layer_accent = LinearLayer(rng = numpy_rng,
                                           input=self.sigmoid_layers[-1].output,
                                           n_in=hidden_layers_sizes[-1],
                                           n_out=n_outs)
               
        elif self.output_activation == 'sigmoid':
            self.final_layer_accent = SigmoidLayer(rng = numpy_rng,
                                         input=self.sigmoid_layers[-1].output,
                                         n_in=hidden_layers_sizes[-1],
                                         n_out=n_outs, activation=T.nnet.sigmoid)
        else:
            print ("This output activation function: %s is not supported right now!" %(self.output_activation))
            sys.exit(1)
            

        self.params.extend(self.final_layer_accent.params)
        self.delta_params.extend(self.final_layer_accent.delta_params)
        
            
        ##MSE FOR EACH SPEAKER           
        self.error_cost = T.mean(T.sum( (self.final_layer_accent.output-self.y)*(self.final_layer_accent.output-self.y), axis=1 ))
        

        ###L1-norm
        if self.l1_reg is not None:
            for i in xrange(self.n_layers):
                W = self.params[i * 2]
                self.error_cost += self.l1_reg * (abs(W).sum())
                
                
        ###L2-norm
        if self.l2_reg is not None:
            for i in xrange(self.n_layers):
                W = self.params[i * 2]
                self.error_cost += self.l2_reg *  T.sqr(W).sum()
                 
                

# <rawcell>

# BUILD FINETUNE FUNCTIONS

# <codecell>

    def build_finetune_functions(self, io_train_set, io_test_set,batch_size):
    
        i_train_set, o_train_set = io_train_set
        i_valid_set, o_valid_set = io_test_set
    
        #compute number of minibatches for training, validation and testing
        n_valid_batches = i_valid_set.get_value(borrow=True).shape[0]
        n_valid_batches /= batch_size
    
        index = T.lscalar('index')  #index to a [mini]batch
        learning_rate = T.fscalar('learning_rate')
        momentum = T.fscalar('momentum')
    
        layer_size = len(self.params)
        lr_list = []
    
        for i in xrange(layer_size):
            lr_list.append(learning_rate)
        
        ##top 2 layers use a smaller learning rate
        if layer_size > 4:
            
            for i in range(4, layer_size):
                #Learning rate for shared hidden layers
                if i < layer_size-4:
                    lr_list[i] = learning_rate * 0.7
                #Learning rate for accent hidden layer    
                elif i < layer_size-2:
                    lr_list[i] = learning_rate * 0.5
                #Learning rate for output layer    
                else:
                    lr_list[i] = learning_rate * 0.5

        #compute list of fine-tuning updates
        #compute the gradients with respect to the model parameters
        gparams = T.grad(self.error_cost,self.params)
        
        updates = theano.compat.python2x.OrderedDict()
        layer_index = 0
            
        for dparam,gparam in zip(self.delta_params,gparams): 
            updates[dparam] = momentum * dparam - gparam * lr_list[layer_index]
            layer_index += 1
              
            
        for dparam,param in zip(self.delta_params,self.params):
            updates[param] = param + updates[dparam]
        
        train_fn = theano.function(inputs = [index,theano.Param(learning_rate,default = 0.0001),
                                             theano.Param(momentum,default =0.5)],
                                             outputs = self.error_cost,
                                             updates = updates,
                                             givens = {self.x: i_train_set[index*batch_size:
                                                                 (index + 1)*batch_size],
                                             self.y: o_train_set[index*batch_size:
                                                                 (index + 1)* batch_size]})
        valid_fn = theano.function([],
                                   outputs=self.error_cost,
                                   givens={self.x: i_valid_set,
                                           self.y: o_valid_set})
        
        valid_score_i = theano.function([index],
                                        outputs = self.error_cost,
                                        givens = {self.x: i_valid_set[index * batch_size:
                                                                      (index + 1) * batch_size],
                                                  self.y: o_valid_set[index *batch_size:
                                                                     (index + 1) * batch_size]})
        
        def valid_score():
            return [valid_score_i(i) for i in xrange(n_valid_batches)]
        
        
        
        
        
        return train_fn, valid_fn



