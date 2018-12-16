from __future__ import print_function
from __future__ import division

import re
import time
import itertools
import numpy as np
import nltk

# For pretty-printing
import pandas as pd
from IPython.display import display, HTML

import sent_segment # py file
from . import constants

nltk.download('treebank') # sentence segmentation

##
# Package and module utils
def require_package(package_name):
    import pkgutil
    import subprocess
    import sys
    if not pkgutil.find_loader(package_name):
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', package_name])

# def run_tests(test_module, test_names, reload=True):
#     import unittest
#     if reload:
#         import importlib
#         importlib.reload(test_module)
#     unittest.TextTestRunner(verbosity=2).run(
#         unittest.TestLoader().loadTestsFromNames(
#             test_names, test_module))

##
# Miscellaneous helpers
def flatten(list_of_lists):
    """Flatten a list-of-lists into a single list."""
    return list(itertools.chain.from_iterable(list_of_lists))

def render_matrix(M, rows=None, cols=None, dtype=float, float_fmt="{0:.04f}"):
    """Render a matrix to HTML using Pandas.

    Args:
      M : 2D numpy array
      rows : list of row labels
      cols : list of column labels
      dtype : data type (float or int)
      float_fmt : format specifier for floats

    Returns:
      (string) HTML representation of M
    """
    df = pd.DataFrame(M, index=rows, columns=cols, dtype=dtype)
    old_fmt_fn = pd.get_option('float_format')
    pd.set_option('float_format', lambda f: float_fmt.format(f))
    html = df._repr_html_()
    pd.set_option('float_format', old_fmt_fn)  # reset Pandas formatting
    return html

def pretty_print_matrix(*args, **kw):
    """Pretty-print a matrix using Pandas.

    Args:
      M : 2D numpy array
      rows : list of row labels
      cols : list of column labels
      dtype : data type (float or int)
      float_fmt : format specifier for floats
    """
    display(HTML(render_matrix(*args, **kw)))


def pretty_timedelta(fmt="%d:%02d:%02d", since=None, until=None):
    """Pretty-print a timedelta, using the given format string."""
    since = since or time.time()
    until = until or time.time()
    delta_s = until - since
    hours, remainder = divmod(delta_s, 3600)
    minutes, seconds = divmod(remainder, 60)
    return fmt % (hours, minutes, seconds)


##
# Word processing functions
def canonicalize_digits(word):
    if any([c.isalpha() for c in word]): return word
    word = re.sub("\d", "DG", word)
    if word.startswith("DG"):
        word = word.replace(",", "") # remove thousands separator
    return word


def canonicalize_word(word, wordset=None, digits=True):
    word = word.lower()
    if digits:
        if (wordset != None) and (word in wordset): return word
        word = canonicalize_digits(word) # try to canonicalize numbers
    if (wordset == None) or (word in wordset):
        return word
    else:
        return constants.UNK_TOKEN

def canonicalize_words(words, **kw):
    return [canonicalize_word(word, **kw) for word in words]

##
# Data loading functions
# def get_corpus(name):
#     import nltk
#     assert(nltk.download(name))
#     return nltk.corpus.__getattr__(name)

def build_vocab(corpus, V=10000, **kw):
    from . import vocabulary
    if isinstance(corpus, list):
        token_feed = (canonicalize_word(w) for w in corpus)
        vocab = vocabulary.Vocabulary(token_feed, size=V, **kw)
    else:
        token_feed = (canonicalize_word(w) for w in corpus.words())
        vocab = vocabulary.Vocabulary(token_feed, size=V, **kw)

    print("Vocabulary: {:,} types".format(vocab.size))
    return vocab

def get_train_test_doc(source,target, split=0.8, shuffle=True):
    """Generate train/test split for unsupervised tasks.

    Args:
      corpus: a list of the article body, each is an array of sencence list
      split (double): fraction to use as training set
      shuffle (int or bool): seed for shuffle of input data, or False to just
      take the training data as the first xx% contiguously.

    Returns:
      train_sentences, test_sentences ( list(list(string)) ): the train and test
      splits
    """
#     sentences = np.array(list(corpus.sents()), dtype=object)

    fmt = (len(source), sum(map(len, source)))
    print("Loaded {:,} documents ({:g} sentences)".format(*fmt)) 

#     if shuffle:
#         rng = np.random.RandomState(shuffle)
#         rng.shuffle(sentences)  # in-place
    split_idx = int(split * len(source))
    train_source = source[:split_idx]
    test_source = source[split_idx:]
       
    train_target = target[:split_idx]
    test_target = target[split_idx:]

    fmt = (len(train_source), sum(map(len, train_source)))
    print("Training set: {:,} articles ({:,} tokens)".format(*fmt))
    fmt = (len(test_source), sum(map(len, test_source)))
    print("Test set: {:,} articles ({:,} tokens)".format(*fmt))

    return train_source, test_source, train_target, test_target



def preprocess_sentences(sentences, vocab, use_eos=False, emit_ids=True,
                         progressbar=lambda l:l):
    """Preprocess sentences by canonicalizing and mapping to ids.

    Args:
      sentences ( list(list(string)) ): input sentences
      vocab: Vocabulary object, already initialized
      use_eos: if true, will add </s> token to end of sentences.
      emit_ids: if true, will emit as ids. Otherwise, will be preprocessed
          tokens.
      progressbar: (optional) progress bar to wrap iterator.

    Returns:
      ids ( array(int) ): flattened array of sentences, including boundary <s>
      tokens.
    """
  
    
    # Add sentence boundaries, canonicalize, and handle unknowns
    word_preproc = lambda w: canonicalize_word(w, wordset=vocab.word_to_id)
    ret = []
    for i in range(len(sentences)):
        for s in progressbar(sentences[i]):
            canonical_words = vocab.pad_sentence(list(map(word_preproc, s)),
                                                 use_eos=use_eos)
            ret.extend(vocab.words_to_ids(canonical_words) if emit_ids else
                       canonical_words)
    if not use_eos:  # add additional <s> to end if needed
        ret.append(vocab.START_ID if emit_ids else vocab.START_TOKEN)
    return np.array(ret, dtype=(np.int32 if emit_ids else object))


def load_data(source, target, split=0.8, V=10000, shuffle=0):
    """Load a data set and split train/test along sentences.

    This is a convenience wrapper to chain together several functions from this
    module, and produce a train/test split suitable for input to most models.

    Sentences are preprocessed by canonicalization and converted to ids
    according to the constructed vocabulary, and interspersed with <s> tokens
    to denote sentence bounaries.

    Args:
        source, target: a list of selected sentecences from extractor model/abstract of the article
        split: (float \in (0,1]) fraction of examples in train split
        V: (int) vocabulary size (including special tokens)
        shuffle: (int) if > 0, use as random seed to shuffle sentence prior to
            split. Can change this to get different splits.

    Returns:
        (vocab, train_x_ids, test_x_ids, train_y_ids, test_y_ids)
        vocab: vocabulary.Vocabulary object
        x_ids: list of list of ids, each item in the inner list represents ids of input sentences of one article
        y_ids: list of list of ids, each item in the inner list represents ids of abstract of one article
    """
    
    # tokenized and get vocabulary
    all_tokens = []
    source_tokens = []
    target_tokens = []
    for i in range(len(source)):
        current_source_tokens= nltk.wordpunct_tokenize(source[i]) # one item represents several sentences
        all_tokens.extend(current_source_tokens)
        source_tokens.append(current_source_tokens)
    for j in range(len(target)):
        current_target_tokens= nltk.wordpunct_tokenize(target[i])
        all_tokens.extend(current_target_tokens)
        target_tokens.append(current_target_tokens)
    vocab = build_vocab(all_tokens, V)
    
    
#     # sentence segmentation of the source/target paragraph
#     paragraph = []
#     for t in range(len(source_tokens)):
#         sentence = sent_segment.segment_sentences(source_tokens[t])
#         paragraph.append(sentence)
      
    
    train_source, test_source, train_target, test_target = get_train_test_doc(source_tokens, target_tokens, split, shuffle)
    
    train_x_ids = []
    test_x_ids = []
    train_y_ids = []
    test_y_ids = []
    for i in range(len(train_source)):
        train_x_ids.append(preprocess_sentences(train_source[i], vocab))
    for i in range(len(test_source)):
        test_x_ids.append(preprocess_sentences(test_source[i], vocab))
    for i in range(len(train_target)):
        train_y_ids.append(preprocess_sentences(train_target[i], vocab))
    for i in range(len(test_target)):
        test_y_ids.append(preprocess_sentences(test_target[i], vocab))
    
    return vocab, train_x_ids, test_x_ids, train_y_ids, test_y_ids

    
##
# Window and batch functions
def pad_np_array(example_ids, max_len=250, pad_id=0):
    """Pad a list of lists of ids into a rectangular NumPy array.

    Longer sequences will be truncated to max_len ids, while shorter ones will
    be padded with pad_id.

    Args:
        example_ids: list(list(int)), sequence of ids for each example
        max_len: maximum sequence length
        pad_id: id to pad shorter sequences with

    Returns: (x, ns)
        x: [num_examples, max_len] NumPy array of integer ids
        ns: [num_examples] NumPy array of sequence lengths (<= max_len)
    """
    arr = np.full([len(example_ids), max_len], pad_id, dtype=np.int32)
    ns = np.zeros([len(example_ids)], dtype=np.int32)
    for i, ids in enumerate(example_ids):
        cpy_len = min(len(ids), max_len)
        arr[i,:cpy_len] = ids[:cpy_len]
        ns[i] = cpy_len
    return arr, ns

def id_lists_to_sparse_bow(id_lists, vocab_size):
    """Convert a list-of-lists-of-ids to a sparse bag-of-words matrix.

    Args:
        id_lists: (list(list(int))) list of lists of word ids
        vocab_size: (int) vocab size; must be greater than the largest word id
            in id_lists.

    Returns:
        (scipy.sparse.csr_matrix) where each row is a sparse vector of word
        counts for the corresponding example.
    """
    from scipy import sparse
    ii = []  # row indices (example ids)
    jj = []  # column indices (token ids)
    for row_id, ids in enumerate(id_lists):
        ii.extend([row_id]*len(ids))
        jj.extend(ids)
    x = sparse.csr_matrix((np.ones_like(ii), (ii, jj)),
                          shape=[len(id_lists), vocab_size])
    return x

def rnnlm_batch_generator(ids, batch_size, max_time):
    """Convert ids to data-matrix form for RNN language modeling."""
    # Clip to multiple of max_time for convenience
    clip_len = ((len(ids)-1) // batch_size) * batch_size
    input_w = ids[:clip_len]     # current word
    target_y = ids[1:clip_len+1]  # next word
    # Reshape so we can select columns
    input_w = input_w.reshape([batch_size,-1])
    target_y = target_y.reshape([batch_size,-1])
    
    # Yield batches
    for i in range(0, input_w.shape[1], max_time):
        yield input_w[:,i:i+max_time], target_y[:,i:i+max_time]

# def rnnlm_batch_generator(x_ids, y_ids, batch_size):
#     """Convert ids to data-matrix form for RNN language modeling.
#      arg: x_ids: list (np.array(int) of ids) ??
#           y_ids: flat (1D) np.array(int) of ids
#      return: encoder_input, decoder_input, decoder_output 
#      [batch_size, max_decoder_time] ??"""
#     for i in range(0, len(x_ids), batch_size):
#         encoder_inputs = x_ids[i:i+batch_size]
# #         if i >0:
#         decoder_inputs = y_ids[i-1:i+batch_size-1]
#         decoder_outputs = y_ids[i:i+batch_size]
#         yield encoder_inputs, decoder_inputs, decoder_outputs


def pad_sentence_batch(sentence_batch, pad_int):
    """Pad sentences with <PAD> so that each sentence of a batch has the same length"""
    max_sentence = max([len(sentence) for sentence in sentence_batch])
    return [sentence + [pad_int] * (max_sentence - len(sentence)) for sentence in sentence_batch]
        
def get_batches(x_ids, y_ids, batch_size, pad_int=1): #END_TOKEN
    """Batch targets, sources, and the lengths of their paragragh together"""
    for batch_i in range(0, len(x_ids)//batch_size):
        start_i = batch_i * batch_size

        # Slice the right amount for the batch
        sources_batch = x_ids[start_i:start_i + batch_size]
        targets_batch = y_ids[start_i:start_i + batch_size]

        # Pad
        pad_sources_batch = np.array(pad_sentence_batch(sources_batch, pad_int))
        pad_targets_batch = np.array(pad_sentence_batch(targets_batch, pad_int))

        # Need the lengths for the _lengths parameters
        pad_targets_lengths = []
        for target in pad_targets_batch:
            pad_targets_lengths.append(len(target))

        pad_source_lengths = []
        for source in pad_sources_batch:
            pad_source_lengths.append(len(source))

        yield pad_sources_batch, pad_targets_batch, pad_source_lengths, pad_targets_lengths
        
        
        
def build_windows(ids, N, shuffle=True):
    """Build window input to the window model.

    Takes a sequence of ids, and returns a data matrix where each row
    is a window and target for the window model. For N=3:
        windows[i] = [w_3, w_2, w_1, w_0]

    For language modeling, N is the context size and you can use y = windows[:,-1]
    as the target words and x = windows[:,:-1] as the contexts.

    For CBOW, N is the window size and you can use y = windows[:,N/2] as the target words
    and x = np.hstack([windows[:,:N/2], windows[:,:N/2+1]]) as the contexts.

    For skip-gram, you can use x = windows[:,N/2] as the input words and y = windows[:,i]
    where i != N/2 as the target words.

    Args:
      ids: np.array(int32) of input ids
      shuffle: if true, will randomly shuffle the rows

    Returns:
      windows: np.array(int32) of shape [len(ids)-N, N+1]
        i.e. each row is a window, of length N+1
    """
    windows = np.zeros((len(ids)-N, N+1), dtype=int)
    for i in range(N+1):
        # First column: first word, etc.
        windows[:,i] = ids[i:len(ids)-(N-i)]
    if shuffle:
        # Shuffle rows
        np.random.shuffle(windows)
    return windows


def batch_generator(data, batch_size):
    """Generate minibatches from data.

    Args:
      data: array-like, supporting slicing along first dimension
      batch_size: int, batch size

    Yields:
      minibatches of maximum size batch_size
    """
    for i in range(0, len(data), batch_size):
        yield data[i:i+batch_size]

def multi_batch_generator(batch_size, *data_arrays):
    """Generate minibatches from multiple columns of data.

    Example:
        for (bx, by) in multi_batch_generator(5, x, y):
            # bx is minibatch for x
            # by is minibatch for y

    Args:
      batch_size: int, batch size
      data_arrays: one or more array-like, supporting slicing along the first
        dimension, and with matching first dimension.

    Yields:
      minibatches of maximum size batch_size
    """
    assert(data_arrays)
    num_examples = len(data_arrays[0])
    for i in range(1, len(data_arrays)):
        assert(len(data_arrays[i]) == num_examples)

    for i in range(0, num_examples, batch_size):
        # Yield matching slices from each data array.
        yield tuple(data[i:i+batch_size] for data in data_arrays)
