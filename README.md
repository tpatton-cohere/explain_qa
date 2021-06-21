# Explainable QA
**Author:** Thomas Patton

## Introduction
The purpose of this repo is to hold all code used in developing an explainable question-answering model. For the time being, only one approach has been stored which is described below.

## Explainable BERT-QA
The code for this project can be found in the ``/bert_qa/`` directory. The goal for this project was to find a way to generate an explanation for a pre-trained BERT question-answering model. An approach similar to the one mentioned [here](https://colab.research.google.com/drive/1tTiOgJ7xvy3sjfiFC9OozbjAX1ho8WN9?usp=sharing) was used. By computing a forward pass-through the BERT model, the gradients of the inputs can be tracked using TensorFlow's GradientTape. The length of these gradients quantify how much a change in the input dimension would affect the output, and therefore the answer. 

All of the code for this was encapsulated into the ``gradients.py`` module with usage as demonstrated in ``gradients_demo.ipynb``. Here, the module is used to compute the gradients for a sample context and query. Though not viewable in GitHub, the responses are exported to an HTML with each word of the input passage highlighted according to its gradient.