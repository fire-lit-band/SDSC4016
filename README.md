# SDSC4016 Project

This is the formal code of our group's project

## Abstract

Matrix factorization (MF) is one of the most common method in the recommender system. And the  idea of using DNN to enhance basic MF model is also a trend in recommender system. Recent works DMF and NCF, which are selected as our study focus, have shown the learning capab-
ility of neural network.  Further, in order to enhance the final results of DMF model, sentiment analysis and the social information are also ensemble into the basic DMF model, which indeed shows a good performance.

## Simple Document

Following is the simple description of the code and the folder. For the detailed explanation, you can click on the title and go deeper to see more detailed explanation

### DMF.py

It is the backbone of our project. Other methods are done based on this code

### [preprocess](preprocess)

The detailed code of cleaning data and train test dataset split

### [ensemble](ensemble)

The implement of our two ensemble models: sentiment ensemble model and social ensemble model

### [sentiment](sentiment)

Code of the implementation of the sentiment analysis

### [baseline](baseline)

The two baseline models that we use to compare: traditional MF model and MF_baseline.ipynb
