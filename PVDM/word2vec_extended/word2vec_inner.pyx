#!/usr/bin/env cython
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# coding: utf-8
#
# Copyright (C) 2013 Radim Rehurek <me@radimrehurek.com>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

import cython
import numpy as np
cimport numpy as np

from libc.math cimport exp
from libc.string cimport memset

from cpython cimport PyCapsule_GetPointer # PyCObject_AsVoidPtr
from scipy.linalg.blas import fblas

REAL = np.float32
ctypedef np.float32_t REAL_t
ctypedef np.uint32_t  INT_t

DEF MAX_SENTENCE_LEN = 1000

ctypedef void (*scopy_ptr) (const int *N, const float *X, const int *incX, float *Y, const int *incY) nogil
ctypedef void (*saxpy_ptr) (const int *N, const float *alpha, const float *X, const int *incX, float *Y, const int *incY) nogil
ctypedef float (*sdot_ptr) (const int *N, const float *X, const int *incX, const float *Y, const int *incY) nogil
ctypedef double (*dsdot_ptr) (const int *N, const float *X, const int *incX, const float *Y, const int *incY) nogil
ctypedef double (*snrm2_ptr) (const int *N, const float *X, const int *incX) nogil
ctypedef void (*sscal_ptr) (const int *N, const float *alpha, const float *X, const int *incX) nogil

ctypedef void (* fast_sentence_sg_ptr) (
    const np.uint32_t *word_point, const np.uint8_t *word_code, const int codelen,
    REAL_t *syn0, REAL_t *syn1, REAL_t *syn0hist, REAL_t *syn1hist, REAL_t *syn0mu, REAL_t *syn1mu, const int size,
    const np.uint32_t word2_index, const REAL_t alpha, INT_t *num_steps, REAL_t *work) nogil

ctypedef void (*fast_sentence_cbow_ptr) (
    const np.uint32_t *word_point, const np.uint8_t *word_code, int codelens[MAX_SENTENCE_LEN],
    REAL_t *neu1,  REAL_t *syn0, REAL_t *syn1, const int size,
    np.uint32_t indexes[MAX_SENTENCE_LEN], const REAL_t alpha, REAL_t *work,
    int i, int j, int k) nogil

cdef scopy_ptr scopy=<scopy_ptr>PyCapsule_GetPointer(fblas.scopy._cpointer , NULL)  # y = x
cdef saxpy_ptr saxpy=<saxpy_ptr>PyCapsule_GetPointer(fblas.saxpy._cpointer , NULL)  # y += alpha * x
cdef sdot_ptr sdot=<sdot_ptr>PyCapsule_GetPointer(fblas.sdot._cpointer     , NULL)      # float = dot(x, y)
cdef dsdot_ptr dsdot=<dsdot_ptr>PyCapsule_GetPointer(fblas.sdot._cpointer  , NULL)   # double = dot(x, y)
cdef snrm2_ptr snrm2=<snrm2_ptr>PyCapsule_GetPointer(fblas.snrm2._cpointer, NULL) # sqrt(x^2)
cdef sscal_ptr sscal=<sscal_ptr>PyCapsule_GetPointer(fblas.sscal._cpointer, NULL) # x = alpha * x
cdef fast_sentence_sg_ptr fast_sentence_sg
cdef fast_sentence_cbow_ptr fast_sentence_cbow


DEF EXP_TABLE_SIZE = 1000
DEF MAX_EXP = 6

cdef REAL_t[EXP_TABLE_SIZE] EXP_TABLE

cdef int ONE = 1
cdef REAL_t ONEF = <REAL_t>1.0

cdef void fast_paragraph_sg_summation(REAL_t *syn0, REAL_t *syn1, REAL_t *syn0vocab, \
                    const int low, const int high, const np.uint32_t *indexes, const int numwords, \
                    const int word_embedding_size, const int paragraph_size, const int logistic_regression_size, \
                    const int paragraph_index, const np.uint32_t *paragraph_points, const np.uint8_t *paragraph_code, \
                    const int codelen, const REAL_t alpha, REAL_t *work) nogil:

    cdef long long b, i
    cdef long long row1 = paragraph_index * paragraph_size, row2
    cdef REAL_t f, g

    # first we reset the update vector for the embedding of a paragraph
    memset(work, 0, paragraph_size * cython.sizeof(REAL_t))

    # for each binary piece of the code of this paragraph, we check how far off we are:
    for b in range(codelen):
        # in the logistic regression matrix each row is stored one after another,
        # so to find the right row corresponding to the right logistic regression
        # in our tree we multiply the index of this regressor by the size of a column:
        row2 = paragraph_points[b] * logistic_regression_size

        # Then we ask ourselves which portions of the regressor:

        # 1. we take care of the paragraph, thus the first 0 up to size of paragraph embed.

        f = <REAL_t>sdot(&paragraph_size, &syn0[row1], &ONE, &syn1[row2], &ONE)

        # 2. we take care of the _summed words_. How many words are they? 2 * window + 1

        for i in range(numwords):
            # we offset by size, but all words _are summed_ so
            # we _do not slide over_ syn1:
            if low+i >= 0 and low + i < high:
                f += <REAL_t>sdot(&word_embedding_size, &syn0vocab[indexes[low + i]*word_embedding_size], &ONE, &syn1[row2+paragraph_size], &ONE)

        # 3. we now fully have f as the result of the paragraph and the words times
        #    the syn1 vector for this regressor.

        if f <= -MAX_EXP or f >= MAX_EXP:
            continue

        f = EXP_TABLE[<int>((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]        

        g = (1 - paragraph_code[b] - f) * alpha

        # update work by the contributions from this gradient step for all values in syn1
        saxpy(&paragraph_size, &g, &syn1[row2], &ONE, work, &ONE)

        # Backprop syn1, syn0 by the error brought upon by input:

        # 1. error due to the paragraph

        saxpy(&paragraph_size, &g, &syn0[row1], &ONE, &syn1[row2], &ONE)

        # 2. error due to the words:

        for i in range(numwords):
            if low+i >= 0 and low + i < high:
                # we offset by size to deal with summed word contributions:
                saxpy(&word_embedding_size, &g, &syn0vocab[indexes[low + i]*word_embedding_size], &ONE, &syn1[row2+paragraph_size], &ONE)
        
    saxpy(&paragraph_size, &ONEF, work, &ONE, &syn0[row1], &ONE)

cdef void fast_paragraph_sg_concatenation(REAL_t *syn0, REAL_t *syn1, REAL_t *syn0vocab, REAL_t *padding_word, \
                    const int low, const int high, const np.uint32_t *indexes, const int window, const int word_embedding_size, const int paragraph_size, const int logistic_regression_size, \
                    const int paragraph_index, const np.uint32_t *paragraph_points, const np.uint8_t *paragraph_code, \
                    const int codelen, const REAL_t alpha, REAL_t *work, REAL_t *padding_word_work) nogil:

    cdef long long a, b, i
    cdef long long row1 = paragraph_index * paragraph_size, row2
    cdef REAL_t f, g

    # Window size
    # -----------
    #
    # Number of words in the window (regardless of whether the
    # actual sentence is longer or shorter):
    cdef int numwords = window * 2

    # Resets
    # ------
    #
    # 1. We reset the update vector for the embedding of a paragraph
    memset(work, 0, paragraph_size * cython.sizeof(REAL_t))

    # 2. We reset the update vector for the padding word:
    memset(padding_word_work, 0, word_embedding_size * cython.sizeof(REAL_t))

    # Binary code criterion
    # ---------------------
    # 
    # for each code piece 'b' we obtain an error for syn1 for this code index
    # and for syn0, and the 'padding word':
    #
    for b in range(codelen):

        # Index of the regressor
        # ----------------------
        #
        # in the logistic regression matrix each row is stored
        # one after another, so to find the right row corresponding
        # to the right logistic regression in our tree we multiply
        # the index of this regressor by the size of a column:
        row2 = paragraph_points[b] * logistic_regression_size

        # Activation of code
        # ------------------
        #
        # Obtain the activation for this binary code, and update
        # the right regressor:
        #
        # 1. we take care of the paragraph dotted with syn1,
        #    thus the first 0...size of paragraph:

        f = <REAL_t>sdot(&paragraph_size, &syn0[row1], &ONE, &syn1[row2], &ONE)

        # 2. then we take care of the concatenated words:
        #    word dotted with syn1:
        for i in range(numwords):
            # we offset by size, and by i * word_embedding_size
            if low+i >= 0 and low + i < high:
                f += <REAL_t>sdot(&word_embedding_size, &syn0vocab[indexes[low+i] * word_embedding_size], &ONE, &syn1[row2+paragraph_size+i*word_embedding_size], &ONE)
            else:
                # concatenate using some other weird device
                f += <REAL_t>sdot(&word_embedding_size, padding_word, &ONE, &syn1[row2+paragraph_size+i*word_embedding_size], &ONE)
        
        if f <= -MAX_EXP or f >= MAX_EXP:
            continue

        # f = <REAL_t>exp(-f) #EXP_TABLE[<int>((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
        # f = 1.0 / ( 1.0 + f)

        f = EXP_TABLE[<int>((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]

        # 3. we now fully have f as the result of the paragraph and the words times
        #    the syn1 vector for this regressor.      

        # Error
        # -----
        #
        # The criterion is the delta between the code we get at 'b'
        # and the one stored in our Huffman tree:
        g = (1 - paragraph_code[b] - f) * alpha


        # Backprop
        # --------
        #
        # 1. gradient for paragraph vector stored in work:
        saxpy(&paragraph_size, &g, &syn1[row2], &ONE, work, &ONE)

        # 2. Regressor's error due to paragraph:

        saxpy(&paragraph_size, &g, &syn0[row1], &ONE, &syn1[row2], &ONE)

        # 3. Regressor's error due to words
        #    and 'padding word' 's error due to regressor:

        for i in range(numwords):
            # we offset by size, and by i * word_embedding_size
            if low+i >= 0 and low + i < high:
                # contribution due to vocabulary word:
                saxpy(&word_embedding_size, &g, &syn0vocab[indexes[low+i] * word_embedding_size], &ONE, &syn1[row2 + paragraph_size + i*word_embedding_size], &ONE)
            else:
                # update 'padding word' work by the contributions from thsi gradient step:
                saxpy(&word_embedding_size, &g, &syn1[row2 + paragraph_size + i * word_embedding_size], &ONE, padding_word_work, &ONE)
                # contribution due to padding word:
                saxpy(&word_embedding_size, &g, padding_word, &ONE, &syn1[row2 + paragraph_size + i * word_embedding_size], &ONE)

    # update paragraph vector:
    saxpy(&paragraph_size, &ONEF, work, &ONE, &syn0[row1], &ONE)
    # update 'padding word':
    saxpy(&word_embedding_size, &ONEF, padding_word_work, &ONE, padding_word, &ONE)


cdef void fast_sentence_sg_original(
    const np.uint32_t *word_point, const np.uint8_t *word_code, const int codelen,
    REAL_t *syn0, REAL_t *syn1, const int size,
    const np.uint32_t word2_index, const REAL_t alpha, REAL_t *work) nogil:

    cdef long long a, b
    cdef long long row1 = word2_index * size, row2
    cdef REAL_t f, g

    memset(work, 0, size * cython.sizeof(REAL_t))
    for b in range(codelen):
        row2 = word_point[b] * size
        f = <REAL_t>sdot(&size, &syn0[row1], &ONE, &syn1[row2], &ONE)

        if f <= -MAX_EXP or f >= MAX_EXP:
            continue

        f = EXP_TABLE[<int>((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]        

        g = (1 - word_code[b] - f) * alpha
        saxpy(&size, &g, &syn1[row2], &ONE, work, &ONE)
        saxpy(&size, &g, &syn0[row1], &ONE, &syn1[row2], &ONE)
        
    saxpy(&size, &ONEF, work, &ONE, &syn0[row1], &ONE)


cdef void fast_sentence_sg_backward_original(
    const np.uint32_t *word_point, const np.uint8_t *word_code, const int codelen,
    REAL_t *syn0forward, REAL_t *syn1forward, REAL_t *syn0backward, REAL_t *syn1backward,
    const int size, const np.uint32_t word2_index, const REAL_t alpha, REAL_t *work) nogil:

    cdef long long a, b
    cdef long long other_word_index = word2_index * size, center_word
    cdef REAL_t f, g

    # going from other_word_index (other word's) forward model

    #    <word 2>  •  •  • <center word>
    #    [forward]           [backward]

    # to current word's backward model.
    memset(work, 0, size * cython.sizeof(REAL_t))
    for b in range(codelen):
        center_word = word_point[b] * size
        f = <REAL_t>sdot(&size, &syn0forward[other_word_index], &ONE, &syn1backward[center_word], &ONE)

        if f <= -MAX_EXP or f >= MAX_EXP:
            continue

        f = EXP_TABLE[<int>((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]        

        g = (1 - word_code[b] - f) * alpha
        saxpy(&size, &g, &syn1backward[center_word], &ONE, work, &ONE)
        saxpy(&size, &g, &syn0forward[other_word_index], &ONE, &syn1backward[center_word], &ONE)
        
    saxpy(&size, &ONEF, work, &ONE, &syn0forward[other_word_index], &ONE)

cdef void fast_sentence_sg_forward_original(
    const np.uint32_t *word_point, const np.uint8_t *word_code, const int codelen,
    REAL_t *syn0forward, REAL_t *syn1forward, REAL_t *syn0backward, REAL_t *syn1backward,
    const int size, const np.uint32_t word2_index, const REAL_t alpha, REAL_t *work) nogil:

    cdef long long a, b
    cdef long long other_word_index = word2_index * size, center_word
    cdef REAL_t f, g

    # going from other_word_index (other word's) backward model

    #    <center word>  •  •  • <word 2>
    #    [forward]              [backward]

    # to current word's forward model.
    memset(work, 0, size * cython.sizeof(REAL_t))
    for b in range(codelen):
        center_word = word_point[b] * size
        f = <REAL_t>sdot(&size, &syn0backward[other_word_index], &ONE, &syn1forward[center_word], &ONE)

        if f <= -MAX_EXP or f >= MAX_EXP:
            continue

        f = EXP_TABLE[<int>((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]        

        g = (1 - word_code[b] - f) * alpha
        saxpy(&size, &g, &syn1forward[center_word], &ONE, work, &ONE)
        saxpy(&size, &g, &syn0backward[other_word_index], &ONE, &syn1forward[center_word], &ONE)
        
    saxpy(&size, &ONEF, work, &ONE, &syn0backward[other_word_index], &ONE)


cdef void fast_sentence0_sg(
    const np.uint32_t *word_point, const np.uint8_t *word_code, const int codelen,
    REAL_t *syn0,     REAL_t *syn1,
    REAL_t *syn0hist, REAL_t *syn1hist,
    REAL_t *syn0mu,   REAL_t *syn1mu,
    const int size,
    const np.uint32_t word2_index,
    const REAL_t alpha,
    INT_t *num_steps,
    REAL_t *work) nogil:

    cdef long long a, b
    cdef long long row1 = word2_index * size, row2
    
    cdef REAL_t f, g, alpha_neg, neg_alpha_divided_by_steps, g_alphaed

    alpha_neg = - alpha

    if num_steps[word2_index] < 1:
        neg_alpha_divided_by_steps = <REAL_t>(-alpha)
    else:
        neg_alpha_divided_by_steps = <REAL_t>(-alpha) / (<REAL_t>(num_steps[word2_index]))

    memset(work, 0, size * cython.sizeof(REAL_t))
    for b in range(codelen):
        row2 = word_point[b] * size
        f = <REAL_t>sdot(&size, &syn0[row1], &ONE, &syn1[row2], &ONE)

        if f <= -MAX_EXP or f >= MAX_EXP:
            continue

        f = EXP_TABLE[<int>((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]        

        g = (1 - word_code[b] - f)

        # create projetion onto syn0:
        saxpy(&size, &g, &syn1[row2], &ONE, work, &ONE)

        # update using history:
        saxpy(&size, &alpha_neg, &syn1hist[row2], &ONE, &syn1[row2], &ONE)

        # update using mean:
        saxpy(&size, &neg_alpha_divided_by_steps, &syn1mu[row2], &ONE, &syn1[row2], &ONE)
        g_alphaed = g * alpha

        # update using current gradient:
        saxpy(&size, &g_alphaed, &syn0[row1], &ONE, &syn1[row2], &ONE)

        # save for later:
        saxpy(&size, &g, &syn0[row1], &ONE, &syn1hist[row2], &ONE)

    saxpy(&size, &alpha, work, &ONE, &syn0[row1], &ONE)

    # # minus previous gradient step:
    saxpy(&size, &alpha_neg, &syn0hist[row1], &ONE, &syn0[row1], &ONE)

    # # plus mean of gradients:
    saxpy(&size, &neg_alpha_divided_by_steps, &syn0mu[row1], &ONE, &syn0[row1], &ONE)

cdef void fast_sentence1_sg(
    const np.uint32_t *word_point, const np.uint8_t *word_code, const int codelen,
    REAL_t *syn0,     REAL_t *syn1,
    REAL_t *syn0hist, REAL_t *syn1hist,
    REAL_t *syn0mu,   REAL_t *syn1mu,
    const int size,
    const np.uint32_t word2_index,
    const REAL_t alpha,
    INT_t *num_steps,
    REAL_t *work) nogil:

    cdef long long a, b
    cdef long long row1 = word2_index * size, row2
    cdef REAL_t f, g, alpha_neg, neg_alpha_divided_by_steps, g_alphaed

    alpha_neg = - alpha

    if num_steps[word2_index] < 1:
        neg_alpha_divided_by_steps = <REAL_t>(-alpha)
    else:
        neg_alpha_divided_by_steps = <REAL_t>(-alpha) / (<REAL_t>(num_steps[word2_index]))

    memset(work, 0, size * cython.sizeof(REAL_t))
    for b in range(codelen):
        row2 = word_point[b] * size
        f = <REAL_t>sdot(&size, &syn0[row1], &ONE, &syn1[row2], &ONE)

        if f <= -MAX_EXP or f >= MAX_EXP:
            continue

        f = EXP_TABLE[<int>((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]        

        g = (1 - word_code[b] - f)

        # create projetion onto syn0:
        saxpy(&size, &g, &syn1[row2], &ONE, work, &ONE)

        # update using history:
        saxpy(&size, &alpha_neg, &syn1hist[row2], &ONE, &syn1[row2], &ONE)

        # update using mean:
        saxpy(&size, &neg_alpha_divided_by_steps, &syn1mu[row2], &ONE, &syn1[row2], &ONE)
        g_alphaed = g * alpha

        # update using current gradient:
        saxpy(&size, &g_alphaed, &syn0[row1], &ONE, &syn1[row2], &ONE)

        # save for later:
        saxpy(&size, &g, &syn0[row1], &ONE, &syn1hist[row2], &ONE)

    saxpy(&size, &alpha, work, &ONE, &syn0[row1], &ONE)

    # # minus previous gradient step:
    saxpy(&size, &alpha_neg, &syn0hist[row1], &ONE, &syn0[row1], &ONE)

    # # plus mean of gradients:
    saxpy(&size, &neg_alpha_divided_by_steps, &syn0mu[row1], &ONE, &syn0[row1], &ONE)

cdef void fast_sentence_sg_svrg(
    const np.uint32_t *word_point, const np.uint8_t *word_code, const int codelen,
    REAL_t *syn0,     REAL_t *syn1,
    REAL_t *syn0hist, 
    REAL_t *syn0mu,  
    const int size,
    const np.uint32_t word2_index,
    const REAL_t alpha,
    INT_t * num_steps,
    REAL_t *work) nogil:

    cdef long long a, b
    cdef long long row1 = word2_index * size, row2
    cdef REAL_t f, g, alpha_neg, neg_alpha_divided_by_steps, g_alphaed

    alpha_neg = - alpha

    if num_steps[word2_index] < 1:
        neg_alpha_divided_by_steps = <REAL_t>(-alpha)
    else:
        neg_alpha_divided_by_steps = <REAL_t>(-alpha) / (<REAL_t>(num_steps[word2_index]))

    memset(work, 0, size * cython.sizeof(REAL_t))
    for b in range(codelen):
        row2 = word_point[b] * size
        f = <REAL_t>sdot(&size, &syn0[row1], &ONE, &syn1[row2], &ONE)

        if f <= -MAX_EXP or f >= MAX_EXP:
            continue

        f = EXP_TABLE[<int>((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]        

        g = (1 - word_code[b] - f)

        # create projetion onto syn0:
        saxpy(&size, &g, &syn1[row2], &ONE, work, &ONE)

        # update using mean:
        g_alphaed = g * alpha

        # update using current gradient:
        saxpy(&size, &g_alphaed, &syn0[row1], &ONE, &syn1[row2], &ONE)

    saxpy(&size, &alpha, work, &ONE, &syn0[row1], &ONE)

    # minus previous gradient step:
    saxpy(&size, &alpha_neg, &syn0hist[row1], &ONE, &syn0[row1], &ONE)

    # plus mean of gradients:
    saxpy(&size, &neg_alpha_divided_by_steps, &syn0mu[row1], &ONE, &syn0[row1], &ONE)


cdef void fast_sentence2_sg(
    const np.uint32_t *word_point, const np.uint8_t *word_code, const int codelen,
    REAL_t *syn0, REAL_t *syn1,
    REAL_t *syn0hist, REAL_t *syn1hist,
    REAL_t *syn0mu,   REAL_t *syn1mu,
    const int size,
    const np.uint32_t word2_index, const REAL_t alpha, 
    INT_t *num_steps, REAL_t *work) nogil:

    cdef long long a, b
    cdef long long row1 = word2_index * size, row2
    cdef REAL_t f, g

    for a in range(size):
        work[a] = <REAL_t>0.0
    for b in range(codelen):
        row2 = word_point[b] * size
        f = <REAL_t>0.0
        for a in range(size):
            f += syn0[row1 + a] * syn1[row2 + a]
        
        if f <= -MAX_EXP or f >= MAX_EXP:
            continue

        f = EXP_TABLE[<int>((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]

        g = (1 - word_code[b] - f) * alpha
        for a in range(size):
            work[a] += g * syn1[row2 + a]
        for a in range(size):
            syn1[row2 + a] += g * syn0[row1 + a]

    for a in range(size):
        syn0[row1 + a] += work[a]


cdef void fast_sentence0_cbow(
    const np.uint32_t *word_point, const np.uint8_t *word_code, int codelens[MAX_SENTENCE_LEN],
    REAL_t *neu1, REAL_t *syn0, REAL_t *syn1, const int size,
    const np.uint32_t indexes[MAX_SENTENCE_LEN], const REAL_t alpha, REAL_t *work,
    int i, int j, int k) nogil:

    cdef long long a, b
    cdef long long row2
    cdef REAL_t f, g, count, inv_count
    cdef int m

    memset(neu1, 0, size * cython.sizeof(REAL_t))
    count = <REAL_t>0.0
    for m in range(j, k):
        if m == i or codelens[m] == 0:
            continue
        else:
            count += ONEF
            saxpy(&size, &ONEF, &syn0[indexes[m] * size], &ONE, neu1, &ONE)
    if count > (<REAL_t>0.5):
        inv_count = ONEF/count
        sscal(&size, &inv_count, neu1, &ONE)

    memset(work, 0, size * cython.sizeof(REAL_t))
    for b in range(codelens[i]):
        row2 = word_point[b] * size
        f = <REAL_t>dsdot(&size, neu1, &ONE, &syn1[row2], &ONE)
        if f <= -MAX_EXP or f >= MAX_EXP:
            continue
        f = EXP_TABLE[<int>((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
        g = (1 - word_code[b] - f) * alpha
        saxpy(&size, &g, &syn1[row2], &ONE, work, &ONE)
        saxpy(&size, &g, neu1, &ONE, &syn1[row2], &ONE)

    for m in range(j, k):
        if m == i or codelens[m] == 0:
            continue
        else:
            saxpy(&size, &ONEF, work, &ONE, &syn0[indexes[m] * size], &ONE)

cdef void fast_sentence1_cbow(
    const np.uint32_t *word_point, const np.uint8_t *word_code, int codelens[MAX_SENTENCE_LEN],
    REAL_t *neu1, REAL_t *syn0, REAL_t *syn1, const int size,
    const np.uint32_t indexes[MAX_SENTENCE_LEN], const REAL_t alpha, REAL_t *work,
    int i, int j, int k) nogil:

    cdef long long a, b
    cdef long long row2
    cdef REAL_t f, g, count, inv_count
    cdef int m

    memset(neu1, 0, size * cython.sizeof(REAL_t))
    count = <REAL_t>0.0
    for m in range(j, k):
        if m == i or codelens[m] == 0:
            continue
        else:
            count += ONEF
            saxpy(&size, &ONEF, &syn0[indexes[m] * size], &ONE, neu1, &ONE)
    if count > (<REAL_t>0.5):
        inv_count = ONEF/count
        sscal(&size, &inv_count , neu1, &ONE)

    memset(work, 0, size * cython.sizeof(REAL_t))
    for b in range(codelens[i]):
        row2 = word_point[b] * size
        f = <REAL_t>sdot(&size, neu1, &ONE, &syn1[row2], &ONE)
        if f <= -MAX_EXP or f >= MAX_EXP:
            continue
        f = EXP_TABLE[<int>((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
        g = (1 - word_code[b] - f) * alpha
        saxpy(&size, &g, &syn1[row2], &ONE, work, &ONE)
        saxpy(&size, &g, neu1, &ONE, &syn1[row2], &ONE)

    for m in range(j, k):
        if m == i or codelens[m] == 0:
            continue
        else:
            saxpy(&size, &ONEF, work, &ONE, &syn0[indexes[m]*size], &ONE)

cdef void fast_sentence2_cbow(
    const np.uint32_t *word_point, const np.uint8_t *word_code, int codelens[MAX_SENTENCE_LEN],
    REAL_t *neu1, REAL_t *syn0, REAL_t *syn1, const int size,
    const np.uint32_t indexes[MAX_SENTENCE_LEN], const REAL_t alpha, REAL_t *work,
    int i, int j, int k) nogil:

    cdef long long a, b
    cdef long long row2
    cdef REAL_t f, g, count
    cdef int m

    for a in range(size):
        neu1[a] = <REAL_t>0.0
    count = <REAL_t>0.0
    for m in range(j, k):
        if m == i or codelens[m] == 0:
            continue
        else:
            count += ONEF
            for a in range(size):
                neu1[a] += syn0[indexes[m] * size + a]
    if count > (<REAL_t>0.5):
        for a in range(size):
            neu1[a] /= count

    for a in range(size):
        work[a] = <REAL_t>0.0
    for b in range(codelens[i]):
        row2 = word_point[b] * size
        f = <REAL_t>0.0
        for a in range(size):
            f += neu1[a] * syn1[row2 + a]
        if f <= -MAX_EXP or f >= MAX_EXP:
            continue
        f = EXP_TABLE[<int>((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
        g = (1 - word_code[b] - f) * alpha
        for a in range(size):
            work[a] += g * syn1[row2 + a]
        for a in range(size):
            syn1[row2 + a] += g * neu1[a]

    for m in range(j, k):
        if m == i or codelens[m] == 0:
            continue
        else:
            for a in range(size):
                syn0[indexes[m] * size + a] += work[a]

def train_sentence_sg_forward(model, sentence, alpha, _work):
    cdef REAL_t *syn0forward      = <REAL_t *>(np.PyArray_DATA(model.syn0forward))
    cdef REAL_t *syn0backward     = <REAL_t *>(np.PyArray_DATA(model.syn0backward))
    cdef REAL_t *syn1forward      = <REAL_t *>(np.PyArray_DATA(model.syn1forward))
    cdef REAL_t *syn1backward     = <REAL_t *>(np.PyArray_DATA(model.syn1backward))

    cdef REAL_t *work
    cdef np.uint32_t word2_index
    cdef REAL_t _alpha = alpha
    cdef int size = model.layer1_size

    cdef np.uint32_t *points[MAX_SENTENCE_LEN]
    cdef np.uint8_t *codes[MAX_SENTENCE_LEN]
    cdef int codelens[MAX_SENTENCE_LEN]
    cdef np.uint32_t indexes[MAX_SENTENCE_LEN]
    cdef np.uint32_t reduced_windows[MAX_SENTENCE_LEN]
    cdef int sentence_len
    cdef int window = model.window

    cdef int i, j, k
    cdef long result = 0

    # convert Python structures to primitive types, so we can release the GIL
    work = <REAL_t *>np.PyArray_DATA(_work)
    sentence_len = <int>min(MAX_SENTENCE_LEN, len(sentence))
    for i in range(sentence_len):
        word = sentence[i]
        if word is None:
            codelens[i] = 0
        else:
            indexes[i]         = word.index
            codelens[i]        = <int>len(word.code)
            codes[i]           = <np.uint8_t *>np.PyArray_DATA(word.code)
            # switch to forward point and rear point:
            points[i]          = <np.uint32_t *>np.PyArray_DATA(word.point)
            result += 1

    # release GIL & train on the sentence
    with nogil:
        for i in range(sentence_len):
            if codelens[i] == 0:
                continue
            j = i - window
            if j < 0:
                j = 0
            k = i + window + 1
            if k > sentence_len:
                k = sentence_len

            # train backwards, up to center word:
            for j in range(j, k):
                if j == i or codelens[j] == 0:
                    continue

                if i > j:
                    fast_sentence_sg_backward_original(points[i], codes[i], codelens[i],
                        syn0forward, syn1forward, syn0backward, syn1backward,
                        size, indexes[j], _alpha, work
                    )
                    # fast_sentence_sg_original(points[i], codes[i], codelens[i],
                    #     syn0backward, syn1backward,
                    #     size, indexes[j], _alpha, work)
                else:
                    fast_sentence_sg_forward_original(points[i], codes[i], codelens[i],
                        syn0forward, syn1forward, syn0backward, syn1backward,
                        size, indexes[j], _alpha, work
                    )
                    # fast_sentence_sg_original(points[i], codes[i], codelens[i],
                    # syn0backward, syn1backward,
                    # size, indexes[j], _alpha, work)
                # fast_sentence_sg_backward_original(points[i], codes[i], codelens[i],
                #     syn0forward, syn1forward, syn0backward, syn1backward,
                #     size, indexes[j], _alpha, work
                # )
                

    return result


def train_sentence_para_pvdm(model, paragraph_object, sentence, alpha, _work, _padding_word_work):
    cdef REAL_t *syn0     = <REAL_t *>(np.PyArray_DATA(model.syn0))
    cdef REAL_t *syn1     = <REAL_t *>(np.PyArray_DATA(model.syn1))


    cdef REAL_t *syn0vocab     = <REAL_t *>(np.PyArray_DATA(model._vocabulary.syn0))

    cdef REAL_t *work

    cdef np.uint32_t word2_index
    cdef REAL_t _alpha = alpha
    cdef int size = model.layer1_size
    cdef int logistic_regression_size = model.logistic_regression_size
    cdef int word_embedding_size = model.word_embedding_size

    cdef np.uint32_t indexes[MAX_SENTENCE_LEN]
    cdef int sentence_len
    cdef int codelen = <int>len(paragraph_object.code)
    cdef int window = model.window

    cdef int i, j, k
    cdef long result = 1

    # PV-DM addition to new binary tree:

    cdef bint concatenate = <bint>model.concatenate

    cdef np.uint32_t paragraph_index   = paragraph_object.index
    cdef np.uint8_t *paragraph_code    = <np.uint8_t *>np.PyArray_DATA(paragraph_object.code)
    cdef np.uint32_t *paragraph_points = <np.uint32_t *>np.PyArray_DATA(paragraph_object.point)

    # Padding word and its associated work array for updating it:
    cdef REAL_t *padding_word  = <REAL_t *>(np.PyArray_DATA(model.padding_word))
    cdef REAL_t *padding_word_work
    padding_word_work          = <REAL_t *>np.PyArray_DATA(_padding_word_work)


    # convert Python structures to primitive types, so we can release the GIL
    work              = <REAL_t *>np.PyArray_DATA(_work)
    

    sentence_len = <int>min(MAX_SENTENCE_LEN, len(sentence))
    for i in range(sentence_len):
        indexes[i] = sentence[i].index

    # release GIL & train on the sentence
    with nogil:
        for i in range(sentence_len):
            j = i - window
            k = i + window + 1

            # chop off nothingness:
            if j < 0:
                j = 0
            if k > sentence_len:
                k = sentence_len

            if not concatenate:

                # perform inference and backprop on work and paragraph:
                fast_paragraph_sg_summation(syn0, syn1, syn0vocab, \
                    j, k, indexes, window, word_embedding_size, size, logistic_regression_size, \
                    paragraph_index, paragraph_points, paragraph_code, codelen, _alpha, work)
            else:
                # build concatenation of words,
                # then use it for inference:

                fast_paragraph_sg_concatenation(syn0, syn1, syn0vocab, padding_word, \
                    j, k, indexes, window, word_embedding_size, size, logistic_regression_size, \
                    paragraph_index, paragraph_points, paragraph_code, codelen, _alpha, work, padding_word_work)


    return result

def train_sentence_sg_original(model, sentence, alpha, _work):
    cdef REAL_t *syn0     = <REAL_t *>(np.PyArray_DATA(model.syn0))
    cdef REAL_t *syn1     = <REAL_t *>(np.PyArray_DATA(model.syn1))

    cdef REAL_t *work
    cdef np.uint32_t word2_index
    cdef REAL_t _alpha = alpha
    cdef int size = model.layer1_size

    cdef np.uint32_t *points[MAX_SENTENCE_LEN]
    cdef np.uint8_t *codes[MAX_SENTENCE_LEN]
    cdef int codelens[MAX_SENTENCE_LEN]
    cdef np.uint32_t indexes[MAX_SENTENCE_LEN]
    cdef np.uint32_t reduced_windows[MAX_SENTENCE_LEN]
    cdef int sentence_len
    cdef int window = model.window

    cdef int i, j, k
    cdef long result = 0

    # convert Python structures to primitive types, so we can release the GIL
    work = <REAL_t *>np.PyArray_DATA(_work)
    sentence_len = <int>min(MAX_SENTENCE_LEN, len(sentence))
    for i in range(sentence_len):
        word = sentence[i]
        if word is None:
            codelens[i] = 0
        else:
            indexes[i] = word.index
            codelens[i] = <int>len(word.code)
            codes[i] = <np.uint8_t *>np.PyArray_DATA(word.code)
            points[i] = <np.uint32_t *>np.PyArray_DATA(word.point)
            reduced_windows[i] = np.random.randint(window)
            result += 1

    # release GIL & train on the sentence
    with nogil:
        for i in range(sentence_len):
            if codelens[i] == 0:
                continue
            j = i - window + reduced_windows[i]
            if j < 0:
                j = 0
            k = i + window + 1 - reduced_windows[i]
            if k > sentence_len:
                k = sentence_len
            for j in range(j, k):
                if j == i or codelens[j] == 0:
                    continue
                fast_sentence_sg_original(points[i], codes[i], codelens[i],
                    syn0, syn1,
                    size, indexes[j], _alpha, work)

    return result

def train_sentence_sg_double_svrg(model, sentence, alpha, _work):
    cdef REAL_t *syn0     = <REAL_t *>(np.PyArray_DATA(model.syn0))
    cdef REAL_t *syn1     = <REAL_t *>(np.PyArray_DATA(model.syn1))
    cdef REAL_t *syn0hist = <REAL_t *>(np.PyArray_DATA(model.syn0hist))
    cdef REAL_t *syn1hist = <REAL_t *>(np.PyArray_DATA(model.syn1hist))
    cdef REAL_t *syn0mu   = <REAL_t *>(np.PyArray_DATA(model.syn0mu))
    cdef REAL_t *syn1mu   = <REAL_t *>(np.PyArray_DATA(model.syn1mu))

    cdef INT_t *syn1stats  = <INT_t *>(np.PyArray_DATA(model.synstats))

    cdef REAL_t *work
    cdef np.uint32_t word2_index
    cdef REAL_t _alpha = alpha
    cdef int size = model.layer1_size

    cdef np.uint32_t *points[MAX_SENTENCE_LEN]
    cdef np.uint8_t *codes[MAX_SENTENCE_LEN]
    cdef int codelens[MAX_SENTENCE_LEN]
    cdef np.uint32_t indexes[MAX_SENTENCE_LEN]
    cdef np.uint32_t reduced_windows[MAX_SENTENCE_LEN]
    cdef int sentence_len
    cdef int window = model.window

    cdef int i, j, k
    cdef long result = 0

    # convert Python structures to primitive types, so we can release the GIL
    work = <REAL_t *>np.PyArray_DATA(_work)
    sentence_len = <int>min(MAX_SENTENCE_LEN, len(sentence))
    for i in range(sentence_len):
        word = sentence[i]
        if word is None:
            codelens[i] = 0
        else:
            indexes[i] = word.index
            codelens[i] = <int>len(word.code)
            codes[i] = <np.uint8_t *>np.PyArray_DATA(word.code)
            points[i] = <np.uint32_t *>np.PyArray_DATA(word.point)
            reduced_windows[i] = np.random.randint(window)
            result += 1

    # release GIL & train on the sentence
    with nogil:
        
        for i in range(sentence_len):
            if codelens[i] == 0:
                continue
            j = i - window + reduced_windows[i]
            if j < 0:
                j = 0
            k = i + window + 1 - reduced_windows[i]
            if k > sentence_len:
                k = sentence_len
            for j in range(j, k):
                if j == i or codelens[j] == 0:
                    continue
                fast_sentence_sg(points[i], codes[i], codelens[i],
                    syn0, syn1,
                    syn0hist, syn1hist,
                    syn0mu, syn1mu,
                    size, indexes[j], _alpha, syn1stats, work)
    return result


def train_sentence_sg_svrg(model, sentence, alpha, _work):
    cdef REAL_t *syn0     = <REAL_t *>(np.PyArray_DATA(model.syn0))
    cdef REAL_t *syn1     = <REAL_t *>(np.PyArray_DATA(model.syn1))
    cdef REAL_t *syn0hist = <REAL_t *>(np.PyArray_DATA(model.syn0hist))
    cdef REAL_t *syn1hist = <REAL_t *>(np.PyArray_DATA(model.syn1hist))
    cdef REAL_t *syn0mu   = <REAL_t *>(np.PyArray_DATA(model.syn0mu))
    cdef REAL_t *syn1mu   = <REAL_t *>(np.PyArray_DATA(model.syn1mu))

    cdef INT_t *syn1stats  = <INT_t *>(np.PyArray_DATA(model.synstats))

    cdef REAL_t *work
    cdef np.uint32_t word2_index
    cdef REAL_t _alpha = alpha
    cdef int size = model.layer1_size

    cdef np.uint32_t *points[MAX_SENTENCE_LEN]
    cdef np.uint8_t *codes[MAX_SENTENCE_LEN]
    cdef int codelens[MAX_SENTENCE_LEN]
    cdef np.uint32_t indexes[MAX_SENTENCE_LEN]
    cdef np.uint32_t reduced_windows[MAX_SENTENCE_LEN]
    cdef int sentence_len
    cdef int window = model.window

    cdef int i, j, k
    cdef long result = 0

    # convert Python structures to primitive types, so we can release the GIL
    work = <REAL_t *>np.PyArray_DATA(_work)
    sentence_len = <int>min(MAX_SENTENCE_LEN, len(sentence))
    for i in range(sentence_len):
        word = sentence[i]
        if word is None:
            codelens[i] = 0
        else:
            indexes[i] = word.index
            codelens[i] = <int>len(word.code)
            codes[i] = <np.uint8_t *>np.PyArray_DATA(word.code)
            points[i] = <np.uint32_t *>np.PyArray_DATA(word.point)
            reduced_windows[i] = np.random.randint(window)
            result += 1

    # release GIL & train on the sentence
    with nogil:
        for i in range(sentence_len):
            if codelens[i] == 0:
                continue
            j = i - window + reduced_windows[i]
            if j < 0:
                j = 0
            k = i + window + 1 - reduced_windows[i]
            if k > sentence_len:
                k = sentence_len
            for j in range(j, k):
                if j == i or codelens[j] == 0:
                    continue
                fast_sentence_sg_svrg(points[i], codes[i], codelens[i],
                    syn0, syn1,
                    syn0hist, 
                    syn0mu, 
                    size, indexes[j], _alpha, syn1stats, work)
    return result

def train_sentence_cbow(model, sentence, alpha, _work, _neu1):
    cdef REAL_t *syn0 = <REAL_t *>(np.PyArray_DATA(model.syn0))
    cdef REAL_t *syn1 = <REAL_t *>(np.PyArray_DATA(model.syn1))

    cdef REAL_t *work
    cdef REAL_t *neu1
    cdef np.uint32_t word2_index
    cdef REAL_t _alpha = alpha
    cdef int size = model.layer1_size

    cdef np.uint32_t *points[MAX_SENTENCE_LEN]
    cdef np.uint8_t *codes[MAX_SENTENCE_LEN]
    cdef int codelens[MAX_SENTENCE_LEN]
    cdef np.uint32_t indexes[MAX_SENTENCE_LEN]
    cdef np.uint32_t reduced_windows[MAX_SENTENCE_LEN]
    cdef int sentence_len
    cdef int window = model.window

    cdef int i, j, k
    cdef long result = 0

    # convert Python structures to primitive types, so we can release the GIL
    work = <REAL_t *>np.PyArray_DATA(_work)
    neu1 = <REAL_t *>np.PyArray_DATA(_neu1)
    sentence_len = <int>min(MAX_SENTENCE_LEN, len(sentence))
    for i in range(sentence_len):
        word = sentence[i]
        if word is None:
            codelens[i] = 0
        else:
            indexes[i] = word.index
            codelens[i] = <int>len(word.code)
            codes[i] = <np.uint8_t *>np.PyArray_DATA(word.code)
            points[i] = <np.uint32_t *>np.PyArray_DATA(word.point)
            reduced_windows[i] = np.random.randint(window)
            result += 1

    # release GIL & train on the sentence
    with nogil:
        for i in range(sentence_len):
            if codelens[i] == 0:
                continue
            j = i - window + reduced_windows[i]
            if j < 0:
                j = 0
            k = i + window + 1 - reduced_windows[i]
            if k > sentence_len:
                k = sentence_len

            fast_sentence_cbow(points[i], codes[i], codelens, neu1, syn0, syn1, size, indexes, _alpha, work, i, j, k)

    return result


def init():
    """
    Precompute function `sigmoid(x) = 1 / (1 + exp(-x))`, for x values discretized
    into table EXP_TABLE.

    """
    global fast_sentence_sg
    global fast_sentence_cbow
    cdef int i
    cdef float *x = [<float>10.0]
    cdef float *y = [<float>0.01]
    cdef float expected = <float>0.1
    cdef int size = 1
    cdef double d_res
    cdef float *p_res

    # build the sigmoid table
    for i in range(EXP_TABLE_SIZE):
        EXP_TABLE[i] = <REAL_t>exp((i / <REAL_t>EXP_TABLE_SIZE * 2 - 1) * MAX_EXP)
        EXP_TABLE[i] = <REAL_t>(EXP_TABLE[i] / (EXP_TABLE[i] + 1))

    # check whether sdot returns double or float
    d_res = dsdot(&size, x, &ONE, y, &ONE)
    p_res = <float *>&d_res
    if (abs(d_res - expected) < 0.0001):
        fast_sentence_sg = fast_sentence0_sg
        fast_sentence_cbow = fast_sentence0_cbow
        return 0  # double
    elif (abs(p_res[0] - expected) < 0.0001):
        fast_sentence_sg = fast_sentence1_sg
        fast_sentence_cbow = fast_sentence1_cbow
        return 1  # float
    else:
        # neither => use cython loops, no BLAS
        # actually, the BLAS is so messed up we'll probably have segfaulted above and never even reach here
        fast_sentence_sg = fast_sentence2_sg
        fast_sentence_cbow = fast_sentence2_cbow
        return 2

FAST_VERSION = init()  # initialize the module
