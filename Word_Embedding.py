# Word Embedding Techniques using Embedding Layer in Keras

import numpy as np
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding

### sentences
sent=[  'the glass of milk',
     'the glass of juice',
     'the cup of tea',
    'I am a good boy',
     'I am a good developer',
     'understand the meaning of words',
     'your videos are good',]

#Vocabulary size
voc_size = 10000

# One Hot Representation
onehot_rep = [one_hot(words , voc_size) for words in sent]
# print(onehot_rep)


# Word Embedding Representation
# for we created a Embedding layer we just convert our sentences to be equal size because Embedding takes equal size sentenses and with the help of pad_sequnces we just convert our sentences.

sent_length = 8
embedded_docs = pad_sequences(onehot_rep , padding="pre" , maxlen = sent_length)
# print(embedded_docs)

dim = 10
model = Sequential()
model.add(Embedding(voc_size , 10, input_length=sent_length))
model.compile("adam" , "mse")

# print(model.summary())

print(model.predict(embedded_docs))