from nltk.stem import WordNetLemmatizer
import nltk


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


with open('generation.txt') as f, open('gen.txt','w') as f1, open('ref.txt','w') as f2:
	while True:
		line = f.readline()
		t = 1
		if line == '':
			break
		if 'mmm' not in line or '. java ' not in line:
			t = 0
		gen = f.readline()
		gen = gen[11:]
		ref = f.readline()
		ref = ref[5:]
		if t == 1:
			# gen = normalize_word(gen)
			f1.write(gen)
			f1.write('\n')
		if t == 1:
			# ref = normalize_word(ref)
			f2.write(ref)
			f2.write('\n')