from preprocessing import preprocess
import numpy as np
import os
from utils import ALLOWED_PUNCTUATION

'''
Returns the word count for a list of tokens
Parameters:
	text: The preprocessed text (list of tokens)
'''
def get_word_count(text):
	return len(text)

'''
Return the character count for a list of tokens
Parameters:
	text: The preprocessed text (list of tokens)
'''
def get_char_count(text):
	return sum([len(word) for word in text])

'''
Returns the average length of tokens in a list
Parameters:
	text: The preprocessed text (list of tokens)
'''
def get_average_word_length(text):
	words_length = [len(word) for word in text]
	return np.mean(words_length)

'''
Returns a list of sentences according to the ALLOWED_PUNCTUATION. Each sentence
is a list of tokens.
Parameters:
	text: The preprocessed text (list of tokens)
Returns:
	sentences: A list of sentences. Each sentence is a list of tokens
'''
def get_sentences(text):

	sentences = []

	# Sliding window
	l = 0
	r = 0
	while(r < len(text)):
		if text[r] not in ALLOWED_PUNCTUATION:
			r += 1
		else:
			sentence = text[l:r]
			sentences.append(sentence)
			r += 1
			l = r
	# Edge case: no punctuation in the end of the last sentence
	if l < r:
		sentences.append(text[l:r])

	return sentences

'''
Returns the number of sentences in the text
Parameters:
	sentences: A list of sentences. Each sentence is a list of tokens
'''
def get_sentence_count(sentences):
	return len(sentences)

'''
Returns the average length of the sentences in the text
Parameters:
	sentences: A list of sentences. Each sentence is a list of tokens
'''
def get_average_sentence_length(sentences):

	lengths = []
	for sentence in sentences:
		lengths.append(len(sentence))
	return np.mean(lengths)

'''
Returns the number of 'biased' words (according to our bias lexicon) present
in the text in question
Parameters:
    text
'''
def get_biased_word_count(text):

	biased_words = _load_bias_lexicon()
	count = sum([text.count(w) for w in biased_words])
	result = float(count)/len(text)
	return result

'''
Loads the bias lexicon according to the path and name
Parameters:
    path: The relative path to the bias lexicon
	file_name: The name of the bias lexicon
'''
def _load_bias_lexicon(path = './lexica/', file_name='bias-lexicon.txt'):
	with open(os.path.join(path, file_name)) as corpus:
		biased_words = corpus.read().split()
	return biased_words




if __name__ == "__main__":

	example_text = "Trump initially seemed to side with Saudi Arabia on the disagreement, but he then instructed Secretary of State Rex Tillerson to back the Kuwaiti mediation initiative. Tillerson and other U.S. diplomats have since traveled through the region to boost Kuwait's efforts, but the dispute has dragged on despite their efforts. Trump was also expected to discuss global efforts to isolate North Korea by halting employment of its guest workers during his talks with Al Sabah. Kuwait has about 6,000 North Korean guest works within its borders as worldwide tensions rise over the Asian nation's pursuit of nuclear weapons. The Associated Press contributed to this report."
	preprocessed = preprocess(example_text)

	sentences = get_sentences(preprocessed)
	for sentence in sentences:
		print(sentence)
	'''
	print(get_word_count(preprocessed))
	print(get_char_count(preprocessed))
	print(get_average_word_length(preprocessed))
	print(get_sentence_count(preprocessed))
	print(get_average_sentence_length(preprocessed))
	'''
