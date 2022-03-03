from nltk import word_tokenize

def lowercase(text):

	for token in text:
		token = token.lowercase()
	return text

def tokenize(text):
	tokens = word_tokenize(text)
	return tokens

def remove_stopwords(text):
	pass
