#!/usr/bin/env python3

from sklearn import preprocessing
from tensorflow.keras.preprocessing.text import Tokenizer

def time_events_to_time_words(dfs):

    X = []
    Y = []
    
    for df in dfs:
        # create event tokens
        #df["sinceMidnight"] = df['datetime'].dt.hour.astype(int) * 3600 + df['datetime'].dt.minute.astype(int) * 60 + df['datetime'].dt.second.astype(int)
        df['minutesSinceMidnight'] = df['datetime'].dt.hour.astype(int) * 60 + df['datetime'].dt.minute.astype(int)
        #df['hoursSinceMidnight'] = df['datetime'].dt.hour.astype(int)
        df['merge'] = df['sensor'].astype(str) + df['value'].astype(str)

        #feature_1 = df["sinceMidnight"].values.astype(str)
        feature_1 = df['minutesSinceMidnight'].values.astype(str)
        #feature_1 = df["hoursSinceMidnight"].values.astype(str)
        feature_2 = df['merge'].values.astype(str)
        
        labels = df['activity'].values.astype(str)

        x_tmp = []
        y_tmp = []

        for i in range(len(feature_1)):
            x_tmp.append(feature_1[i])
            x_tmp.append(feature_2[i])

            # double the Y because the two features had the same label as they are part of the same event
            y_tmp.append(labels[i])
            y_tmp.append(labels[i])

        x_tmp = np.array(x_tmp).astype(str)
        y_tmp = np.array(y_tmp).astype(str)
        
        X.append(x_tmp)
        Y.append(y_tmp)
        
    return X,Y


def time_events_to_time_words(dfs, custom_dict = None):

    le = preprocessing.LabelEncoder()

    raw_sentences, rax_labels = time_events_to_time_words(dfs)
    
    
    
    le.fit(rax_labels)


    le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    # if no dict is provided, gerere a dict
    if custom_dict == None:

        sentences = []
        for raw_sentence in raw_sentences:
            sentence = " ".join(raw_sentence)
            
            sentences.append(sentence)

        tokenizer = Tokenizer(filters='', lower=False)
        tokenizer.fit_on_texts(sentences)

        wordDict = tokenizer.word_index
    else:
        wordDict = custom_dict


    #encode words with dict for each sentence
    
    encoded_sentences = []
    
    for sentence in sentences:
        
        encoded_sentence = [wordDict[word] for word in sentence]
        
        encoded_sentences.append(encoded_sentence)

    X = np.array(encoded_sentences).astype(int)

    # save de word dictionary
    self.eventDict = wordDict