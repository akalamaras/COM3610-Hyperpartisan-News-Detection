

from nltk import word_tokenize
from nltk.corpus import stopwords



def tokenize(text):
	tokens = word_tokenize(text)
	return tokens

def lowercase(tokens_list):

	for token in tokens_list:
		token = token.lower()
	return tokens_list

def remove_stopwords(tokens_list):

	stopwords_list = set(stopwords.words('english'))

	filtered_tokens = []
	for token in tokens_list:
		if token not in stopwords_list:
			filtered_tokens.append(token)

	return filtered_tokens

def preprocess(text):


	tokenized = tokenize(text)
	lowercased = lowercase(tokenized)
	return remove_stopwords(lowercased)
