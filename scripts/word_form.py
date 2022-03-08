from nltk.stem import WordNetLemmatizer
import nltk

# Input: a complete sentence for easy part-of-speech analysis
sentence = 'Adding ConsumerResource and ConsumerPool for ConsumeKafka'
def normalize_word(sentence):
    lemmatizer = WordNetLemmatizer()
    
    tokens = nltk.word_tokenize(sentence)
    pos_tags = nltk.pos_tag(tokens)
    res = []
    for word, pos in pos_tags:
        if pos.startswith('J'):
            tmp = lemmatizer.lemmatize(word.lower(), 'a')
        elif pos.startswith('V'):
            tmp = lemmatizer.lemmatize(word.lower(), 'v')
        elif pos.startswith('N'):
            tmp = lemmatizer.lemmatize(word.lower(), 'n')
        elif pos.startswith('R'):
            tmp = lemmatizer.lemmatize(word.lower(), 'r')
        else:
            tmp = lemmatizer.lemmatize(word.lower())
        res.append(tmp)
    return ' '.join(res)

sentence = normalize_word(sentence)
print(sentence)