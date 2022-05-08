# Hyperpartisan News Detection

Created as part of COM3610: Dissertation Project for the __University of Sheffield__, Department of Computer Science.  

Author: Alexandros Kalamaras  
Supervisor: Nikolaos Aletras

The origin of the idea for the project lies with the project supervisor, Nikolaos Aletras. 
The software is copyrighted in my name and that of the University of Sheffield.

The datasets for training and testing the models developed herein were taken from PAN-SemEval-Hyperpartisan-News-Detection-19.  
**Link:** https://webis.de/data/pan-semeval-hyperpartisan-news-detection-19.html

## Abstract

Politics is an unavoidable part of any modern-day citizen. With the invention of the
internet and the rapid growth of social media, hyper-partisanship is now more widespread
than ever before. Being able to see through these biases is necessary if we want a chance at
arriving at the truth behind political events.   

The aim of this dissertation project is to utilise machine learning techniques in order to
produce a functional classification model which is able to detect hyperpartisan argumentation
in written news articles. It is important to stress that the solution is **not** a mechanism to pinpoint 
the exact political leanings of the author, but rather a binary categorization of the text itself 
based on any present hyper-partisan language.

   
**Keywords**: Hyperpartisanship, Preprocessing, Feature Extraction, Natural
Language Processing, Supervised Machine Learning, Word Embeddings, Recurrent Neural
Networks, Transformers.



### Instructions

All the necessary dependencies for the project to run as expected are in the **requirements.txt** file.  
The classifiers can be run by calling **main.py** via the Command Line.

By default, all models will be trained/fine-tuned and evaluated. Any trained Doc2Vec models will be saved in the **models** directory.

1. These models include:
   * A Doc2Vec model
   * A Doc2Vec model, enhanced with hand-picked features (located in **features.py**)
   * A pre-trained BERT model, which will be fine-tuned.
2. If you wish to not run one or more of the classifier(s), be sure to comment out the specified line(s) in **main.py**
3. If you wish to use the Doc2Vec model you have already trained, be sure to comment out the specified line in **main.py** to avoid retraining the model.
4. For the BERT model, evaluation happens during training/fine-tuning, so the classifier needs to be fine-tuned every time.
