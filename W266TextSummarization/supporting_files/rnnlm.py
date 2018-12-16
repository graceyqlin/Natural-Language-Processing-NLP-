# questions: train_loss, projection layer, target weight


from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import time

import tensorflow as tf
import numpy as np


def matmul3d(X, W):
    """Wrapper for tf.matmul to handle a 3D input tensor X.
    Will perform multiplication along the last dimension.

    Args:
      X: [m,n,k]
      W: [k,l]

    Returns:
      XW: [m,n,l]
    """
    Xr = tf.reshape(X, [-1, tf.shape(X)[2]])
    XWr = tf.matmul(Xr, W)
    newshape = [tf.shape(X)[0], tf.shape(X)[1], tf.shape(W)[1]]
    return tf.reshape(XWr, newshape)


def process_decoder_input(target_data, batch_size):
    """
    Preprocess target data for decoding
    :return: Preprocessed target data
    """
    # get '<GO>' id: 3; <STOP>: 4
    go_id = 0
    # extracts a slice of size (end-begin)/stride from the given input_ tensor. 
    # two dimention??
    after_slice = tf.strided_slice(target_data, [0, 0], [batch_size, -1], [1, 1])# input, begin, end, strides
    filled_go_id = tf.fill([batch_size, 1], go_id)
    after_concat = tf.concat( [filled_go_id, after_slice], 1)
    
    return after_concat


def MakeFancyRNNCell(H, keep_prob, num_layers=1):
    """Make a fancy RNN cell.

    Use tf.nn.rnn_cell functions to construct an LSTM cell.
    Initialize forget_bias=0.0 for better training.

    Args:
      H: hidden state size
      keep_prob: dropout keep prob (same for input and output)
      num_layers: number of cell layers

    Returns:
      (tf.nn.rnn_cell.RNNCell) multi-layer LSTM cell with dropout
    """
    cells = []
    for _ in range(num_layers):
        cell = tf.nn.rnn_cell.BasicLSTMCell(H, forget_bias=0.0)
        #cell = tf.nn.rnn_cell.DropoutWrapper(
        #  cell, input_keep_prob=keep_prob, output_keep_prob=keep_prob)
        cells.append(cell)

    return tf.nn.rnn_cell.MultiRNNCell(cells)


# Decorator-foo to avoid indentation hell.
# Decorating a function as:
# @with_self_graph
# def foo(self, ...):
#     # do tensorflow stuff
#
# Makes it behave as if it were written:
# def foo(self, ...):
#     with self.graph.as_default():
#         # do tensorflow stuff
#
# We hope this will save you some indentation, and make things a bit less
# error-prone.
def with_self_graph(function):
    def wrapper(self, *args, **kwargs):
        with self.graph.as_default():
            return function(self, *args, **kwargs)
    return wrapper


class RNNLM(object):
    def __init__(self, graph=None, *args, **kwargs):
        """Init function.

        This function just stores hyperparameters. You'll do all the real graph
        construction in the Build*Graph() functions below.

        Args:
          V: vocabulary size
          H: hidden state dimension = embedding size
          num_layers: number of RNN layers (see tf.nn.rnn_cell.MultiRNNCell)
        """
        # Set TensorFlow graph. All TF code will work on this graph.
        self.graph = graph or tf.Graph()
        self.SetParams(*args, **kwargs)

    @with_self_graph
    def SetParams(self, V, H, softmax_ns=200, num_layers=1, batch_size=100):
        # Model structure; these need to be fixed for a given model.
        self.V = V
        self.H = H
        self.num_layers = num_layers
        self.batch_size_ = batch_size
        
        self.start_of_sequence_id = 0 # Go
        self.end_of_sequence_id = 1 # END_TOKEN


        # Training hyperparameters; these can be changed with feed_dict,
        # and you may want to do so during training.
        with tf.name_scope("Training_Parameters"):
#             # Number of samples for sampled softmax.
#             self.softmax_ns = softmax_ns

            self.learning_rate_ = tf.placeholder(tf.float32, [], name="learning_rate")

            # For gradient clipping, if you use it.
            # Due to a bug in TensorFlow, this needs to be an ordinary python
            # constant instead of a tf.constant.
            self.max_grad_norm_ = 1.0

            self.use_dropout_ = tf.placeholder_with_default(
                False, [], name="use_dropout")

            # If use_dropout is fed as 'True', this will have value 0.5.
            self.dropout_keep_prob_ = tf.cond(
                self.use_dropout_,
                lambda: tf.constant(0.5),
                lambda: tf.constant(1.0),
                name="dropout_keep_prob")

            # Dummy for use later.
            self.no_op_ = tf.no_op()


    @with_self_graph
    def BuildCoreGraph(self):
        """Construct the core RNNLM graph, needed for any use of the model.

        This should include:
        - Placeholders for input tensors (encoder_inputs_, initial_h_, decoder_outputs_)
        - Variables for model parameters
        - Tensors representing various intermediate states
        - A Tensor for the final state (final_h_)
        - A Tensor for the output logits (logits_), i.e. the un-normalized argument
          of the softmax(...) function in the output layer.
        - A scalar loss function (loss_)

        Your loss function should be a *scalar* value that represents the
        _average_ loss across all examples in the batch (i.e. use tf.reduce_mean,
        not tf.reduce_sum).

        """
        ########################### Define Input Tensors #######################################
        # Input ids, with dynamic shape depending on input. Sourse input words
        # Should be shape [batch_size, paragraph_length], assuming we flattern the list for input and target
        self.encoder_inputs_ = tf.placeholder(tf.int32, [None, None], name="encoder_inputs")  
 
        # target output words, these are decoder_inputs shifted to the left by one time step with an 
        # end-of-sentence tag appended on the right
            # Should be shape [batch_size, paragraph_length], assuming we flattern the list for input and target
        self.decoder_targets_ = tf.placeholder(tf.int32, [None, None], name="decoder_targets") 

        # self.target_sequence_length_ = tf.placeholder(tf.int32, [None], name='target_sequence_length') #batch_size
        self.target_sequence_length_ = tf.reduce_sum(tf.to_int32(tf.not_equal(self.decoder_targets_, 1)), 1)  # 1D vector (reduced to "1" dimension)
        # the length of the longest paragraph from the source input data, used for GreedyEmbeddingHelper
        self.max_target_length_ = tf.reduce_max(self.target_sequence_length_) 


     

        ########################### Encoder Layer #############################################
        # Construct RNN/LSTM cell and recurrent layer.
        with tf.name_scope("Encoder_Layer"):
            # encoder embedding: maps a sequence of symbols to a sequence of embeddings
            #[batch_size, paragraph_length] --> [batch_size, paragraph_length, embed_dim]
            self.encoder_emb_inp_ = tf.contrib.layers.embed_sequence(self.encoder_inputs_, 
                                             vocab_size=self.V, 
                                             embed_dim=self.H)
            # RNN modeling
            self.encoder_cell_ = MakeFancyRNNCell(H=self.H, keep_prob=0, num_layers=self.num_layers)       
            # is initialized correct and necessary??
            self.encoder_initial_ = self.encoder_cell_.zero_state(self.batch_size_, dtype=tf.float32)
            
            #   encoder_outputs: [batch_size, paragraph_length, H]
            #   encoder_final: [batch_size, H]
            self.encoder_outputs_, self.encoder_final_ = tf.nn.dynamic_rnn(
                                           cell=self.encoder_cell_, inputs=self.encoder_emb_inp_,
                                           initial_state=self.encoder_initial_, dtype=tf.float32)
    
        
        ########################### Decoder Layer #############################################  
        # decode training and inference share parameters and variables, should in the same variable scope
        # the number of RNN layers in the decoder model has to be equal to the number of RNN layers in the encoder model
        
        with tf.name_scope("Decoder_Layer"):
            
             ######## Decoder Embedding ############
            self.embedding_ = tf.Variable(tf.random_uniform([self.V, self.H],-1, 1), name="embedding")    
            
            # 2nd argument needs to be the batch size, matching the decoder_initial_ size.
            self.tmp_dec_inp = process_decoder_input(self.decoder_targets_,self.batch_size_)  

            # decoder training and inference share the same embedding parameters
            self.decoder_emb_inp_ = tf.nn.embedding_lookup(self.embedding_, self.tmp_dec_inp)

            
            ######## Decoder Training ############
            self.decoder_cell_ = MakeFancyRNNCell(H=self.H, keep_prob=self.dropout_keep_prob_, num_layers=self.num_layers)
#             self.decoder_cell_ = tf.nn.rnn_cell.BasicLSTMCell(self.H)

#             # Below needs to match dimention of the self.tmp_dec_inp that is also used in the helper. 
#             self.decoder_initial_ = self.decoder_cell_.zero_state(self.batch_size_, dtype=tf.float32)  
    
            # Helper, pass the embedded input to the basic decoder
            self.helper_ = tf.contrib.seq2seq.TrainingHelper(
                self.decoder_emb_inp_, 
                self.target_sequence_length_
                )
    
            # output layer, turn the top hidden states to logit vectors of dimension V
            self.output_layer_ = tf.layers.Dense(self.V, use_bias=False) 
            
            # basic decoder model, connecting the decoder RNN layers and the input prepared by TrainingHelper
            # Decoder, accessing to the source information through initializing it with the last hidden state of the encoder
            self.decoder_ = tf.contrib.seq2seq.BasicDecoder(cell=self.decoder_cell_, helper=self.helper_, 
                            initial_state=self.encoder_final_, output_layer=self.output_layer_ )
            
            # Dynamic decoding, returns (final_outputs, final_state, final_sequence_lengths)
            # output containing training logits and sample_id
            self.outputs_, _ ,_ = tf.contrib.seq2seq.dynamic_decode(self.decoder_,impute_finished=True,
                                                                   maximum_iterations=self.max_target_length_) 
                
            self.logits_ = tf.identity(self.outputs_.rnn_output, name="logits")    


            
             ######## Decoder Inference ############      
            self.decoder_cell_inf_ = MakeFancyRNNCell(H=self.H, keep_prob=self.dropout_keep_prob_, num_layers=self.num_layers)
#             self.decoder_cell_inf_ = tf.nn.rnn_cell.BasicLSTMCell(self.H)

            # provide the decode embedding parameter from training to inference helper
            self.helper_inf_ = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                self.decoder_emb_inp_, 
                tf.fill([self.batch_size_],self.start_of_sequence_id), # id of Go
                self.end_of_sequence_id # if of EOS
                )
                       
            self.decoder_inf_ = tf.contrib.seq2seq.BasicDecoder(cell=self.decoder_cell_inf_, helper=self.helper_inf_, 
                            initial_state=self.encoder_final_, output_layer=self.output_layer_)
            
            # output contains inference logits and sample_id
            self.outputs_inf_, _ ,_ = tf.contrib.seq2seq.dynamic_decode(self.decoder_inf_,impute_finished=True,
                                                                       maximum_iterations=self.max_target_length_) 
     
            self.logits_inf_ = tf.identity(self.outputs_inf_.sample_id, name="predictions")

     

    @with_self_graph
    def BuildTrainGraph(self):
        """Construct the training ops.

        - train_step_ : a training op that can be called once per batch

        """
       
        with tf.name_scope("Optimization"): 
            # use weighted softmax cross entropy loss function
            self.masks_ = tf.sequence_mask(self.target_sequence_length_, self.max_target_length_, dtype=tf.float32, name='masks')
            self.loss_ = tf.contrib.seq2seq.sequence_loss(self.logits_,  self.decoder_targets_, self.masks_)

            self.optimizer_ = tf.train.AdamOptimizer(self.learning_rate_)
        
            # Gradient Clipping
            self.gradients_ = self.optimizer_.compute_gradients(self.loss_)
            self.capped_gradients_ = [(tf.clip_by_value(grad, -1.,1.), var) for grad, var in self.gradients_ if grad is not None]
            self.train_step_ = self.optimizer_.apply_gradients(self.capped_gradients_)

        # Initializer step
        init_ = tf.global_variables_initializer()

