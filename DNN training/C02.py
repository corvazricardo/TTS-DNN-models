# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>



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


from C01 import VCTK_feat_collection
#from dnn import DNN
import matplotlib.pyplot as plt
#import logging 
#import logging.config
import StringIO

#from mlp import HiddenLayer, MLP
#from logistic_sgd import LogisticRegression
from layers import LinearLayer, HiddenLayer, SigmoidLayer

#from theano.sandbox.rng_mrg import MRG_RandomStreams
from theano.tensor.shared_randomstreams import RandomStreams

theano.config.floatX = 'float32'


# <rawcell>

# CREATE DNN OBJECT

# <codecell>

class DNN(object):


    def __init__(self, numpy_rng = numpy.random.RandomState(2**30), theano_rng=None, n_ins=601,
                 n_outs=259, l1_reg = None, l2_reg = None, 
                 hidden_layers_sizes= [256, 256, 256, 256, 256], 
                 hidden_activation='tanh', output_activation='sigmoid'):
        
        print "DNN Initialisation"
        #logger = logging.getLogger("DNN initialization")

        self.sigmoid_layers = []
        self.params = []
        self.delta_params   = []
        self.n_layers = len(hidden_layers_sizes)
        
        self.n_ins = n_ins
        self.n_outs = n_outs
        #self.speaker_ID = []
        
        self.output_activation = output_activation

        self.l1_reg = l1_reg
        self.l2_reg = l2_reg       
        #vctk_class = Code_01.VCTK_feat_collection()
        
        assert self.n_layers > 0
        
        if not theano_rng:
            theano_rng = RandomStreams(numpy.random.randint(2 ** 30))

        # allocate symbolic variables for the data
        self.x = T.matrix('x') 
        self.y = T.matrix('y') 
        
        
        for i in xrange(self.n_layers):
            if i == 0:
                input_size = n_ins
            else:
                input_size = hidden_layers_sizes[i - 1]

            if i == 0:
                layer_input = self.x
            else:
                layer_input = self.sigmoid_layers[-1].output

            sigmoid_layer = HiddenLayer(rng=numpy_rng,
                                        input=layer_input,
                                        n_in=input_size,
                                        n_out=hidden_layers_sizes[i],
                                        activation=T.tanh)  ##T.nnet.sigmoid)  # 
           
           
            self.sigmoid_layers.append(sigmoid_layer)
            self.params.extend(sigmoid_layer.params) 
            self.delta_params.extend(sigmoid_layer.delta_params)
         
     
        # add final layer
        if self.output_activation == 'linear':
            self.final_layer = LinearLayer(rng = numpy_rng,
                                           input=self.sigmoid_layers[-1].output,
                                           n_in=hidden_layers_sizes[-1],
                                           n_out=n_outs)
            
        elif self.output_activation == 'sigmoid':
            self.final_layer = SigmoidLayer(
                 rng = numpy_rng,
                 input=self.sigmoid_layers[-1].output,
                 n_in=hidden_layers_sizes[-1],
                 n_out=n_outs, activation=T.nnet.sigmoid)
        else:
            print ("This output activation function: %s is not supported right now!" %(self.output_activation))
            sys.exit(1)

        self.params.extend(self.final_layer.params)
        self.delta_params.extend(self.final_layer.delta_params)
    
        ### MSE
        self.finetune_cost = T.mean(T.sum( (self.final_layer.output-self.y)*(self.final_layer.output-self.y), axis=1 ))
        
        self.errors = T.mean(T.sum( (self.final_layer.output-self.y)*(self.final_layer.output-self.y), axis=1 ))
        
        ### L1-norm
        if self.l1_reg is not None:
            for i in xrange(self.n_layers):
                W = self.params[i * 2]
                self.finetune_cost += self.l1_reg * (abs(W).sum())

        ### L2-norm
        if self.l2_reg is not None:
            for i in xrange(self.n_layers):
                W = self.params[i * 2]
                self.finetune_cost += self.l2_reg * T.sqr(W).sum()  
        

# <rawcell>

# BUILD FINETUNE FUNCTIONS

# <codecell>

    def build_finetune_functions(self, io_train_set, io_test_set, batch_size):
        
        (i_train_set, o_train_set) = io_train_set
        (i_valid_set, o_valid_set) = io_test_set
        
        # compute number of minibatches for training, validation and testing
        n_valid_batches = i_valid_set.get_value(borrow=True).shape[0]
        n_valid_batches /= batch_size
        
        index = T.lscalar('index')  # index to a [mini]batch
        learning_rate = T.fscalar('learning_rate')
        momentum = T.fscalar('momentum')
        
        layer_size = len(self.params)
        lr_list = []
        for i in xrange(layer_size):
            lr_list.append(learning_rate)

        ##top 2 layers use a smaller learning rate
        if layer_size > 4:
            for i in range(layer_size-4, layer_size):
                lr_list[i] = learning_rate * 0.5

        # compute list of fine-tuning updates
        # compute the gradients with respect to the model parameters
        gparams = T.grad(self.finetune_cost, self.params)

        updates = theano.compat.python2x.OrderedDict()
        layer_index = 0
        
        for dparam, gparam in zip(self.delta_params, gparams):
            updates[dparam] = momentum * dparam - gparam * lr_list[layer_index]
            layer_index += 1

        for dparam, param in zip(self.delta_params, self.params):
            updates[param] = param + updates[dparam]
            
        train_fn = theano.function(inputs=[index, theano.Param(learning_rate, default = 0.0001),
              theano.Param(momentum, default = 0.5)],
              outputs=self.errors,
              updates=updates,
              givens={self.x: i_train_set[index * batch_size:
                                          (index + 1) * batch_size],
                      self.y: o_train_set[index * batch_size:
                                          (index + 1) * batch_size]})
                                          
        valid_fn = theano.function([], 
              outputs=self.errors,
              givens={self.x: i_valid_set,
                      self.y: o_valid_set})

        valid_score_i = theano.function([index], 
              outputs=self.errors,
              givens={self.x: i_valid_set[index * batch_size:
                                          (index + 1) * batch_size],
                      self.y: o_valid_set[index * batch_size:
                                          (index + 1) * batch_size]})
        # Create a function that scans the entire validation set
        def valid_score():
            return [valid_score_i(i) for i in xrange(n_valid_batches)]

        return train_fn, valid_fn
        
        
        
        
        

# <rawcell>

# PARAMETER PREDICTION

# <codecell>

    def parameter_prediction(self, i_test_set):  #, batch_size

        n_test_set_i = i_test_set.get_value(borrow=True).shape[0]

        test_out = theano.function([], self.final_layer.output,
              givens={self.x: i_test_set[0:n_test_set_i]})

        predict_parameter = test_out()

        return predict_parameter

# <rawcell>

# GENERATE TOP HIDDEN LAYER

# <codecell>

    def generate_top_hidden_layer(self, i_test_set, bn_layer_index):
        
        n_test_set_i = i_test_set.get_value(borrow=True).shape[0]

        test_out = theano.function([], self.sigmoid_layers[bn_layer_index].output,
              givens={self.x: i_test_set[0:n_test_set_i]})

        predict_parameter = test_out()

        return predict_parameter

# <codecell>

    def train_DNN(self, i_data, i_frame, o_data, o_frame, n_ins, n_outs, speaker_ID):
    
        bin_class = Code_01.VCTK_feat_collection()
        
        print "Starting train_DNN"
        
        ####Setting parameters####  
        training_epochs=100
        #batch_size = 32
        batch_size = 256
        l1_reg = 0.0 #L1_regularization
        l2_reg = 0.00001 #L2_regularization
        private_l2_reg = 0.00001
        warmup_epoch = 5
        warmup_epoch = 10
        momentum = 0.9
        warmup_momentum = 0.3
        hidden_layers_sizes = [256, 256, 256, 256, 256]
        #hidden_layers_sizes = [1024, 1024, 1024, 1024, 1024, 1024]
        #hidden_layers_sizes = [1024, 1024, 1024, 1024, 1024]
        stream_weights = [1.0]
        private_hidden_sizes = [1024]
        buffer_utt_size = 400
        early_stop_epoch = 5
        hidden_activation = 'tanh'
        output_activation = 'linear'
        #stream_lr_weights
        #use_private_hidden
        #model_type 
        n_ins = 601
        n_outs = 259
        #learning_rate = 0.012
        learning_rate = 0.0002
        finetune_lr = learning_rate
        buffer_size = 200000
        self.speaker_ID= speaker_ID
        print "Speaker ID :", self.speaker_ID
    
        ###MAYBE INCLUDE SOME PRETRAINING####
    
        buffer_size = int(buffer_size / batch_size) * batch_size
    
        
        ###Loading Data###
         
        i_train, o_train, i_test, o_test, train_frame, test_frame = bin_class.make_train_test_data(i_data,
                                                                                                   o_data,
                                                                                                   i_frame,
                                                                                                   o_frame)

        io_shared_train_set, i_temp_train_set, o_temp_train_set = bin_class.load_next_partition(i_train,
                                                                                               o_train, 
                                                                                               train_frame)
    
        io_shared_test_set, i_temp_test_set, o_temp_test_set = bin_class.load_next_partition(i_test,
                                                                                               o_test, 
                                                                                               test_frame)
        
        i_temp_train_set.set_value(numpy.asarray(i_temp_train_set.eval(), dtype = theano.config.floatX), borrow = True)
        o_temp_train_set.set_value(numpy.asarray(o_temp_test_set.eval(), dtype = theano.config.floatX), borrow = True)
        
        # numpy random generator
        numpy_rng = numpy.random.RandomState(123)
    
        dnn_model = None
        train_fn = None
        valid_fn = None
 
        dnn_model = DNN(numpy_rng=numpy_rng, n_ins=n_ins, n_outs = n_outs,
                        l1_reg = l1_reg, l2_reg = l2_reg, 
                         hidden_layers_sizes = hidden_layers_sizes, 
                          hidden_activation = hidden_activation, 
                          output_activation = output_activation)
        
        train_fn, valid_fn = dnn_model.build_finetune_functions(io_shared_train_set, io_shared_test_set, batch_size)
        
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
            

            while (not bin_class.is_finish()):  
                print "Loading new partition"
            
                io_shared_train_set, i_temp_train_set, o_temp_train_set = bin_class.load_next_partition(i_train,
                                                                                               o_train, 
                                                                                               train_frame)
                
                i_temp_train_set.set_value(numpy.asarray(i_temp_train_set.eval(), dtype = theano.config.floatX), borrow = True)
                o_temp_train_set.set_value(numpy.asarray(o_temp_test_set.eval(), dtype = theano.config.floatX), borrow = True)
                
                n_train_batches = i_temp_train_set.get_value().shape[0] / batch_size
                
                for minibatch_index in xrange(n_train_batches):
                    this_train_error = train_fn(minibatch_index, current_finetune_lr, current_momentum)
                    train_error.append(this_train_error)
                
                    if numpy.isnan(this_train_error):
                        print "train_error over minibatch ",minibatch_index+1, "of", n_train_batches, "was", this_train_error
                        #print "training error over minibatch %d of %d was %s", (minibatch_index+1,n_train_batches,this_train_error)
                    
               
            bin_class.reset()
            
            
            print('calculating validation loss')
            validation_losses = valid_fn()
            this_validation_loss = numpy.mean(validation_losses)
            print ("Old validation loss :"), this_validation_loss
            
            if this_validation_loss < best_validation_loss:
                best_dnn_model = dnn_model
                best_validation_loss = this_validation_loss
                print('validation loss decreased, so saving model')
                
            if this_validation_loss >= previous_loss:
                print('validation loss increased')
                dbn = best_dnn_model
                early_stop += 1
            
            if early_stop > early_stop_epoch:
                print('stopping early')
                break
            
            if math.isnan(this_validation_loss):
                break
            
            previous_loss = this_validation_loss
            print ("New validation loss :"), this_validation_loss
        end_time = time.clock()
    
        #cPickle.dump(best_dnn_model, open(nnets_file_name, 'wb'))
    
        print "overall  training time  :", (end_time - start_time) / 60., "validation error", best_validation_loss
        print "Architecture :",hidden_layers_sizes
        '''
        if plot:
            plotlogger.save_plot('training convergence',title='Final training and validation error',xlabel='epochs',ylabel='error')
        '''
        
        return  best_validation_loss, best_dnn_model
           

