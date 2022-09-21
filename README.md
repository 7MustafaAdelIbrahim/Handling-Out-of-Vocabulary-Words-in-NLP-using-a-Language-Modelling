### Handling-Out-of-Vocabulary-Words-in-NLP-using-a-Language-Modelling
          Building a model to produce embeddings for Out-of-Vocabulary(OV) words depending on its context.
          Using a language model built using a bi-directional RNN with a LSTM cell.


Word Embeddings encode the relationships between words through vector representations of the words. These word vectors are analogous to the meaning of the word. A limitation of word embeddings are that, they are learned by the Natural Language Model (word2vec, GloVe and the like) and therefore words must have been seen in the training data before, in order to have an embedding.

This articles provides an approach that can be used to handle out-of-vocabulary(OOV) words in natural language processing. Given an OOV word and the sentence it is in, language modelling is used to sequence words in the sentence and predict the meaning of the word by comparison with similar sentences. This is an elegant way of learning word meanings on the fly.

Overview of Model
The model is built to produce embeddings for Out-of-Vocabulary(OOV) words depending on the OOV word’s context. This is done using language model built using a Bi-directional Recurrent Neural Network with a Long-Short Term Memory cell. This language model is used for predicting the most probable word embedding for the OOV word based on its context, by predicting words in place of the OOV word and then taking a weighted average of their mapped word embeddings. This gives a word embedding for the OOV word which is reliable in terms of usability for entity recognition tasks and a meaningful representation for it in the vector space.

The Model for predicting OOV word embeddings consist of two segments: The first step is where the Model is prepared by tokenising, training and saving a model based on a training corpus, and the second step, which consists of the Prepared model being used to predict embeddings. The first step is called the Preparation Step and the second step is called the Embedding Prediction step.


Model Preparation Step
In Step 1 of the model preparation, shown above, a large corpus is pre-processed to be used for Step 2, where this corpus is tokenised, and the text is encoded as integers.

The tokenizer in Step 2 is used to fit the source text to develop the mapping from words to unique integers. Then these integers can be used to sequence lines of text. The size of the vocabulary of the text corpus is retrieved using the tokenizer to define the word embedding layer of the model inn Step 3.1.1. In Step 2.4, we prepare forward and backward sequences of the text data, where both the directions of the sequences are used for predicting embedding based on the previous and later words from the OOV word. The sequences are then split into input (X) and output elements (y) for each forward and reverse sequences to be used for training them on the RNN LSTM model prediction model. The forward and backward sequences are padded to have all sequences of the same length.

In Step 3, we define a model for each forward and backward sequences with 3 layers. The first embedding layer creates a real valued vector for each sequence. The second layer is the Bi-directional LSTM layer with a set number of units, which can be pruned to best fit the training corpus. The output layer is a dense layer comprised of one neuron for each word in the vocabulary. This layer uses a SoftMax function to ensure that the output is normalised to return a probability.

In Step 4, the encoded text for forward and backward sequences are compiled fit on respective RNNs. Since the Network is technically used to predict the probability distribution of the vocabulary based on the sequence provided, we use a sparse categorical cross entropy loss function to update weights on the network. The Adam optimiser is an efficient implementation of the gradient decent algorithm that is used to track the accuracy of each epoch of the training. The models is then saved to be used to return probability distributions of the vocabulary depending on forward or backward sequenced models.

OOV Embedding Prediction Step

Predict Word Embedding Step
The OOV Word Embedding Prediction step is shorter than the Model preparation step. Step 1, consists of loading all the models and parameters required to run the Embedding Prediction functions. In Step 2, the Generate Sequence function is used as a function that is called by Step 4’s Set Embedding function to be able to predict the most probable words that occur in place of the OOV word in the Sample text. These predictions are used to map to GloVe vectors of the predicted words and the Step 4 function defines a weighted average of the predicted words’ embeddings. This Embedding is assigned to the vocabulary of our pre-trained model. This method of assigning embeddings is used so that, the OOV words will have a reasonable position in the vector space based on its context, even though it was initially not assigned an embedding.

ResearchGate:
https://www.researchgate.net/publication/335757797_Language_Modelling_for_Handling_Out-of-Vocabulary_Words_in_Natural_Language_Processing?showFulltext=1&linkId=5d7a26a04585151ee4afb0c5 
