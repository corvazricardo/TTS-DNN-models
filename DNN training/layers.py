#Code provided by Dr. Zhizheng Wu
#Author: Dr. Zhizheng Wu


import numpy, time, cPickle, gzip, sys, os, copy

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

import logging


class LinearLayer(object):
    def __init__(self, rng, input, n_in, n_out, W = None, b = None):

        self.input = input

        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        if W is None:
            W_value = rng.normal(0.0, 1.0/numpy.sqrt(n_in), size=(n_in, n_out))
            W = theano.shared(value=numpy.asarray(W_value, dtype=theano.config.floatX), name='W', borrow=True)

        if b is None:
            b = theano.shared(value=numpy.zeros((n_out,),
                                        dtype=theano.config.floatX),
                                   name='b', borrow=True)
        
        self.W = W
        self.b = b

        self.delta_W = theano.shared(value = numpy.zeros((n_in,n_out),
                                     dtype=theano.config.floatX), name='delta_W')

        self.delta_b = theano.shared(value = numpy.zeros_like(self.b.get_value(borrow=True),
                                     dtype=theano.config.floatX), name='delta_b')

        self.output = T.dot(self.input, self.W) + self.b

        self.params = [self.W, self.b]
        self.delta_params = [self.delta_W, self.delta_b]

    def errors(self, y):
        L = T.sum( (self.output-y)*(self.output-y), axis=1 )
        errors = T.mean(L)
        return (errors)

    def init_params(self, iparams):
        updates = {}
        for param, iparam in zip(self.params, iparams):
            updates[param] = iparam
        return updates

class SigmoidLayer(object):
    def __init__(self, rng, input, n_in, n_out, W = None, b = None, activation = T.tanh):

        self.input = input

        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        if W is None:
            W_value = numpy.asarray(rng.normal(0.0, 1.0/numpy.sqrt(n_in),
                      size=(n_in, n_out)), dtype=theano.config.floatX)
            W = theano.shared(value=W_value,
                              name='W', borrow=True)
        if b is None:
            b = theano.shared(value=numpy.zeros((n_out,),
                              dtype=theano.config.floatX),
                              name='b', borrow=True)
        
        self.W = W
        self.b = b

        self.delta_W = theano.shared(value = numpy.zeros((n_in,n_out),
                                     dtype=theano.config.floatX), name='delta_W')

        self.delta_b = theano.shared(value = numpy.zeros_like(self.b.get_value(borrow=True),
                                     dtype=theano.config.floatX), name='delta_b')

        self.output = T.dot(self.input, self.W) + self.b
        self.output = activation(self.output)
        
        self.params = [self.W, self.b]
        self.delta_params = [self.delta_W, self.delta_b]

    def errors(self, y):
        L = T.sum( (self.output-y)*(self.output-y), axis=1 )
        errors = T.mean(L)
        return (errors)

    def init_params(self, iparams):
        updates = {}
        for param, iparam in zip(self.params, iparams):
            updates[param] = iparam
        return updates
        

class GeneralLayer(object):

    def __init__(self, rng, input, n_in, n_out, W = None, b = None, activation = 'linear'):
 
        self.input = input
        self.n_in = n_in
        self.n_out = n_out

        self.logger = logging.getLogger('general_layer')        

        # randomly initialise the activation weights based on the input size, as advised by the 'tricks of neural network book'        
        if W is None:
            W_values = numpy.asarray(rng.normal(0.0, 1.0/numpy.sqrt(n_in),
                    size=(n_in, n_out)), dtype=theano.config.floatX)
            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        self.delta_W = theano.shared(value = numpy.zeros((n_in,n_out),
                                     dtype=theano.config.floatX), name='delta_W')

        self.delta_b = theano.shared(value = numpy.zeros_like(self.b.get_value(borrow=True),
                                     dtype=theano.config.floatX), name='delta_b')

        lin_output = T.dot(input, self.W) + self.b
        if activation == 'sigmoid':
            self.output = T.nnet.sigmoid(lin_output)

        elif activation == 'tanh':
            self.output = T.tanh(lin_output)

        elif activation == 'linear':
            self.output = lin_output

        elif activation == 'ReLU':  ## rectifier linear unit
            self.output = T.maximum(0.0, lin_output)

        elif activation == 'ReSU':  ## rectifier smooth unit
            self.output = numpy.log(1.0 + numpy.exp(lin_output))

        else:
            self.logger.critical('the input activation function: %s is not supported right now. Please modify layers.py to support' % (activation))
            raise

        # parameters of the model

        self.params = [self.W, self.b]
        self.delta_params = [self.delta_W, self.delta_b]
    
    def errors(self, y):
        errors = T.mean(T.sum((self.output-y)**2, axis=1))

        return errors

    def init_params(self, iparams):
        updates = {}
        for param, iparam in zip(self.params, iparams):
            updates[param] = iparam
        return updates


class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,
                 activation=T.tanh, do_maxout = False, pool_size = 1,
                 do_pnorm = False, pnorm_order = 1):
        """ Class for hidden layer """
        self.input = input
        self.n_in = n_in
        self.n_out = n_out
        
        if W is None:

            W_values = numpy.asarray(rng.normal(0.0, 1.0/numpy.sqrt(n_in),
                    size=(n_in, n_out)), dtype=theano.config.floatX)

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        self.delta_W = theano.shared(value = numpy.zeros((n_in,n_out),
                                     dtype=theano.config.floatX), name='delta_W')

        self.delta_b = theano.shared(value = numpy.zeros_like(self.b.get_value(borrow=True),
                                     dtype=theano.config.floatX), name='delta_b')

        lin_output = T.dot(input, self.W) + self.b
        if do_maxout == True:
            self.last_start = n_out - pool_size
            self.tmp_output = lin_output[:,0:self.last_start+1:pool_size]
            for i in range(1, pool_size):
                cur = lin_output[:,i:self.last_start+i+1:pool_size]
                self.tmp_output = T.maximum(cur, self.tmp_output)
            self.output = activation(self.tmp_output)
        elif do_pnorm == True:
            self.last_start = n_out - pool_size
            self.tmp_output = abs(lin_output[:,0:self.last_start+1:pool_size]) ** pnorm_order
            for i in range(1, pool_size):
                cur = abs(lin_output[:,i:self.last_start+i+1:pool_size]) ** pnorm_order
                self.tmp_output = self.tmp_output + cur
            self.tmp_output = self.tmp_output ** (1.0 / pnorm_order)
            self.output = activation(self.tmp_output)
        else:
            self.output = (lin_output if activation is None
                           else activation(lin_output))

#        self.output = self.rectifier_linear(lin_output)
        
        # parameters of the model
        self.params = [self.W, self.b]
        self.delta_params = [self.delta_W, self.delta_b]
    
    def rectifier_linear(self, x):
        x = T.maximum(0.0, x)
        
        return  x
        
    def rectifier_smooth(self, x):
        x = numpy.log(1.0 + numpy.exp(x))
        
        return  x

class dA(object):
    def __init__(self, numpy_rng, theano_rng = None, input = None, 
                 n_visible= None, n_hidden= None, W = None, bhid = None,
                 bvis = None, firstlayer = 0, variance   = None ):

        self.n_visible = n_visible
        self.n_hidden  = n_hidden

        # create a Theano random generator that gives symbolic random values
        if not theano_rng : 
            theano_rng = RandomStreams(numpy_rng.randint(2**30))

        if not W:
            initial_W = numpy.asarray( numpy_rng.uniform( 
                      low  = -4*numpy.sqrt(6./(n_hidden+n_visible)), 
                      high =  4*numpy.sqrt(6./(n_hidden+n_visible)), 
                      size = (n_visible, n_hidden)),
                                       dtype = theano.config.floatX)
            W = theano.shared(value = initial_W, name ='W')

        if not bvis:
            bvis = theano.shared(value = numpy.zeros(n_visible, 
                                         dtype = theano.config.floatX))

        if not bhid:
            bhid = theano.shared(value = numpy.zeros(n_hidden,
                                dtype = theano.config.floatX), name ='b')


        self.W = W
        self.b = bhid
        self.b_prime = bvis
        self.W_prime = self.W.T 
        self.theano_rng = theano_rng

        if input == None : 
            self.x = T.dmatrix(name = 'input') 
        else:
            self.x = input

        self.params = [self.W, self.b, self.b_prime]

        # first layer, use Gaussian noise
        self.firstlayer = firstlayer

        if self.firstlayer == 1 :
            if variance == None :
                self.var = T.vector(name = 'input')
            else :
                self.var = variance
        else :
            self.var = None

    def get_corrupted_input(self, input, corruption_level):
        if self.firstlayer == 0 :
            return  self.theano_rng.binomial(
                             size = input.shape, 
                             n = 1, 
                             p = 1 - corruption_level, 
                             dtype=theano.config.floatX) * input
        else :
            noise = self.theano_rng.normal( size = input.shape, 
                                            dtype = theano.config.floatX)
            denoises = noise * self.var * corruption_level
            return input+denoises
    
    def get_hidden_values(self, input):
        return T.nnet.sigmoid(T.dot(input, self.W) + self.b)

    def get_reconstructed_input(self, hidden ):
        if self.firstlayer == 1 :
            return T.dot(hidden, self.W_prime) + self.b_prime
        else :
            return T.nnet.sigmoid(T.dot(hidden, self.W_prime) + self.b_prime)
    
    def get_cost_updates(self, corruption_level, learning_rate):
        tilde_x = self.get_corrupted_input(self.x, corruption_level)
        y       = self.get_hidden_values( tilde_x )
        z       = self.get_reconstructed_input(y)

        L = T.sum ( (self.x-z) * (self.x-z), axis=1 ) 
        cost = T.mean(L) / 2

        gparams = T.grad(cost, self.params)
        updates = {}
        for param, gparam in zip(self.params, gparams):
            updates[param] = param -  learning_rate*gparam
    
        return (cost, updates)

    def init_params(self, iparams):
        updates = {}
        for param, iparam in zip(self.params, iparams):
            updates[param] = iparam
        return updates

    def get_test_cost(self, corruption_level):
        """ This function computes the cost and the updates for one trainng
        step of the dA """
        
        # tilde_x = self.get_corrupted_input(self.x, corruption_level, 0.5)
        y       = self.get_hidden_values( self.x )
        z       = self.get_reconstructed_input(y)
        L = T.sum ( (self.x-z) * (self.x-z), axis=1)
        cost = T.mean(L)

        return cost
