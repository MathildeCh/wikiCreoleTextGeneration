#!/usr/bin/env python
# coding: utf-8

# #### Ici le code divise juste le texte en tokens. Le texte utilisé est le wikicreole de la page wikipedia des Marches

# In[1]:


import spacy


# In[ ]:





# In[ ]:





# In[2]:


def read_file(filepath):
    
    with open(filepath) as f:
        str_text = f.read()
    
    return str_text


# In[3]:


read_file("/home/diego/Desktop/Master_TAL/Corsi/M1_S1_Machine creativity/Project/Marches.txt")


# In[4]:


import spacy
nlp = spacy.load('fr_core_news_sm',disable=['parser', 'tagger','ner'])


# In[5]:


def separate_punc(doc_text):
    return [token.text.lower() for token in nlp(doc_text)]


# In[6]:


d = read_file("/home/diego/Desktop/Master_TAL/Corsi/M1_S1_Machine creativity/Project/Marches.txt")
tokens = separate_punc(d)


# In[7]:


tokens


# In[8]:


len(tokens)


# In[ ]:





# In[ ]:





# #### Ici on divise le texte en séquences de tokens. Cela semble nécessaire au model pour s'entrainer.

# In[9]:


# organize into sequences of tokens
train_len = 25+1 # 50 training words , then one target word

# Empty list of sequences
text_sequences = []

for i in range(train_len, len(tokens)):
    
    # Grab train_len# amount of characters
    seq = tokens[i-train_len:i]
    
    # Add to list of sequences
    text_sequences.append(seq)


# In[116]:


# Exemple d'une séquence

' '.join(text_sequences[0])


# In[11]:


len(text_sequences)


# In[ ]:





# In[ ]:





# #### Fase du preprocessing. Ici on:
# 
# ##### 1) Va assigner un nombre à chaque unique token dans notre texte
# ##### 2) On va re-écrire les séquences avec le numéro qu'on a assigné à chaque token

# In[12]:


from keras.preprocessing.text import Tokenizer


# In[13]:


# integer encode sequences of words
tokenizer = Tokenizer()
tokenizer.fit_on_texts(text_sequences)
sequences = tokenizer.texts_to_sequences(text_sequences)


# In[117]:


# Exemple avec la première séquence où les tokens sont dévenus des nombres

sequences[0]


# In[118]:


# Voici le dico avec tous les token et leur numéro respectif

tokenizer.index_word


# In[16]:


for i in sequences[0]:
    print(f'{i} : {tokenizer.index_word[i]}')


# In[119]:


tokenizer.word_counts


# In[120]:


vocabulary_size = len(tokenizer.word_counts)


# In[121]:


vocabulary_size


# In[ ]:





# In[ ]:





# #### Ici on transforme les séquences de numéros en arrays

# In[20]:


import numpy as np


# In[21]:


sequences = np.array(sequences)


# In[22]:


sequences


# In[ ]:





# In[ ]:





# In[ ]:


#### Ici on construit le modèle et on ajoute les chouches de néurones. Sache que c'est un modèle qui avait fonctionné 
#### mais je ne comprends pas tout au 100% 
#### Certains valeurs sont par défault comme l'activation, l'optimizer.


# In[23]:


import keras
from keras.models import Sequential
from keras.layers import Dense,LSTM,Embedding


# In[122]:


def create_model(vocabulary_size, seq_len):
    # Type de modèle
    model = Sequential()
    # Chaque model.add ajoute une chouche de néurones.
    model.add(Embedding(vocabulary_size, 25, input_length=seq_len))
    model.add(LSTM(50, return_sequences=True))
    model.add(LSTM(50))
    model.add(Dense(50, activation='relu'))

    model.add(Dense(vocabulary_size, activation='softmax'))
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
   
    model.summary()
    
    return model


# In[ ]:





# In[ ]:





# #### Ici on va créer le X (le corpus de données avec lequel le modèle va s'entrainer) et le y (le corpus de données avec lequel le modèle va confronter ses resultats)

# In[25]:


from tensorflow.keras.utils import to_categorical


# In[26]:


sequences


# In[27]:


# First 49 words
sequences[:,:-1]


# In[28]:


# last Word
sequences[:,-1]


# In[29]:


X = sequences[:,:-1]


# In[30]:


y = sequences[:,-1]


# In[31]:


y = to_categorical(y, num_classes=vocabulary_size+1)


# In[32]:


seq_len = X.shape[1]


# In[33]:


seq_len


# In[34]:


# define model
model = create_model(vocabulary_size+1, seq_len)


# In[ ]:





# In[ ]:





# #### Ici on va entrainer le modèle
# 
# ##### ATTENTION! ça prend du temps quand meme...

# In[35]:


# fit model
model.fit(X, y, batch_size=100, epochs=40,verbose=1)


# In[ ]:





# In[ ]:





# #### Ici on devrait etre capable de générer du texte. Mais...

# In[38]:


from random import randint
from pickle import load
from keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


# In[ ]:


random_seed_text = text_sequences[random_pick]


# ##### La fonction "predict_classes" (celle entouré de #### ci-bas) n'existe plus depuis 2021 (function deprecated sur tensorflow)
# ##### J'ai cherché sur internet par quoi la substituer et j'ai trouvé, sauf que la nouvelle ligne de code fait surgir un autre erreur!
# ##### ce dernier erreur est pour moi insourmontable... ça fait 2 heures que j'essaie mais rien.
# ##### Je suis censé convertir un np.ndarray en string mais une fois cela fait on se retrouve avec un problème d'index. Et la c'est perdu.

# In[125]:


def generate_text(model, tokenizer, seq_len, seed_text, num_gen_words):
    '''
    INPUTS:
    model : model that was trained on text data
    tokenizer : tokenizer that was fit on text data
    seq_len : length of training sequence
    seed_text : raw string text to serve as the seed
    num_gen_words : number of words to be generated by model
    '''
    
    # Final Output
    output_text = []
    
    # Intial Seed Sequence
    input_text = seed_text
    
    # Create num_gen_words
    for i in range(num_gen_words):
        
        # Take the input text string and encode it to a sequence
        encoded_text = tokenizer.texts_to_sequences([input_text])[0]
        
        # Pad sequences to our trained rate (50 words in the video)
        pad_encoded = pad_sequences([encoded_text], maxlen=seq_len, truncating='pre')
        
        
        ################################################################
        # Predict Class Probabilities for each word
        pred_word_ind = model.predict_classes(pad_encoded, verbose=0)[0]
        ################################################################
        
        # Grab word
        pred_word = tokenizer.index_word[pred_word_ind] 
        
        # Update the sequence of input text (shifting one over with the new word)
        input_text += ' ' + pred_word
        
        output_text.append(pred_word)
        
    # Make it look like a sentence.
    return ' '.join(output_text)


# In[63]:


text_sequences[0]


# In[ ]:


import random
random.seed(101)
random_pick = random.randint(0,len(text_sequences))


# In[64]:


random_seed_text = text_sequences[random_pick]


# In[65]:


random_seed_text


# In[66]:


seed_text = ' '.join(random_seed_text)


# In[67]:


seed_text


# In[127]:


# c'est ici que le code ne marche plus. Et c'est dommage car c'est la dernière étape...

generate_text(model,tokenizer,seq_len,seed_text=seed_text,num_gen_words=50)

