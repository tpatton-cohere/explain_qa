# GRADIENTS.PY - This module contains a few functions to compute gradient-based explanations for question-answering models
#
# Author: Thomas Patton (thomas.patton@coherehealth.com)
# (c) 2021 Cohere Health

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import torch
from IPython.core.display import display, HTML
from transformers import AutoTokenizer, TFBertForQuestionAnswering

def mask_scores(scores, best_idx, token_ids_size):
    """
    Unhappy this needs to be its own function
    """
    mask = np.zeros((1, token_ids_size))
    mask[0, best_idx] = 1
    mask_tensor = tf.constant(mask, dtype='float32')
    masked_scores = scores * mask_tensor
    return masked_scores


def get_gradients(model, tokenizer, question, context, verbose=False):
    """
    Runs a forward-pass of a question-answering model and returns the gradients of the 
    output w.r.t. the input. This gradient quantifies how much a change in the input dimension
    would change the output.
    
    parameters:
    model (transformers.TFBertForQuestionAnswering) - the QA model to use
    tokenizer (transformers.AutoTokenizer) - the QA tokenizer
    question (string) - the "question" string to feed into the QA model
    context (string) - the context to search for the answer in. no more than 512 characters
    verbose (bool) - whether or not to print verbose output
    
    returns:
    gradient_df (pd.DataFrame) - a dataframe containing information about the words, their gradients, and their token type
    """
    embedding_matrix = model.get_input_embeddings()
    encoded_tokens =  tokenizer.encode_plus(question, context, add_special_tokens=True, return_token_type_ids=True, return_tensors="tf")
    token_ids = list(encoded_tokens["input_ids"].numpy()[0])
    vocab_size = embedding_matrix.vocab_size
    
    # convert token ids to one hot. We can't differentiate wrt to int token ids hence the need for one hot representation
    token_ids_tensor = tf.constant([token_ids], dtype='int32')
    token_ids_tensor_one_hot = tf.one_hot(token_ids_tensor, vocab_size) 
    
    with tf.GradientTape() as tape:
        # (i) watch input variable
        tape.watch(token_ids_tensor_one_hot)

        # multiply input model embedding matrix; allows us do backprop wrt one hot input 
        inputs_embeds = tf.matmul(token_ids_tensor_one_hot, embedding_matrix.weights[0]) 
        
        # (ii) get prediction
        output = model({"inputs_embeds": inputs_embeds, "token_type_ids": encoded_tokens["token_type_ids"], "attention_mask": encoded_tokens["attention_mask"] })
        start_scores = output.start_logits
        end_scores = output.end_logits
        
        # (iii) find the best start and end. this is the "selection" of the QA model
        best_start = np.argmax(start_scores)
        best_end = np.argmax(end_scores) + 1
        
        mask_start_scores = mask_scores(start_scores, best_start, len(token_ids))
        mask_end_scores = mask_scores(end_scores, best_end, len(token_ids))
        
        # (iv) compute and normalize gradients
        gradients = tf.norm(tape.gradient([mask_start_scores, mask_end_scores], token_ids_tensor_one_hot), axis=2).numpy()[0]
        normalized_gradients = gradients / np.max(gradients)
        
        token_words = tokenizer.convert_ids_to_tokens(token_ids)
        answer_text = tokenizer.decode(token_ids[best_start:best_end])
        token_types = list(encoded_tokens["token_type_ids"].numpy()[0])
        
        # (v) organize results into a dataframe
        gradient_df = pd.DataFrame({'word' : token_words,
                                    'gradient' : normalized_gradients,
                                    'type' : token_types})
        
        return gradient_df
    

def html(words, gradients):
    """
    Given a list of words and a list of gradients, returns an string in HTML format with
    the words highlighted according to their gradient
    """
    ret_str = ''
    for i in range(len(words)):
        word = words[i]
        grad_f = gradients[i] * 0.7 + 0.05
        ret_str += f'<span style="background-color: hsla(110, 70%, 50%, {grad_f});">{word}</span><span> </span>'
        
    return ret_str