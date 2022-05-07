

from nltk import word_tokenize
from nltk.corpus import stopwords
from utils import ALLOWED_PUNCTUATION
import re
import html

'''
This method was taken from Yeh et. al. (Semeval 2019 Task 4 Submission)
https://github.com/chialun-yeh/SemEval2019/blob/4cf5b57960100a41943cbba60d7413b0bab100fd/utils.py
'''
def _fix(text):

	text = text.replace('#39;', "'").replace('amp;', '&').replace('#146;', "'").replace('nbsp;', ' ') \
	.replace('#36;', '$').replace('\\n', "\n").replace('quot;', "'").replace('<br />', "\n").replace('\\"', '"')\
	.replace('<unk>','u_n').replace(' @.@ ','.').replace(' @-@ ','-').replace('\\', ' \\ ')
	return html.unescape(text)

'''
This method was taken from Yeh et. al. (Semeval 2019 Task 4 Submission)
https://github.com/chialun-yeh/SemEval2019/blob/4cf5b57960100a41943cbba60d7413b0bab100fd/utils.py

Normalizes all quotations in the given text
'''
def clean_quotations(text):
	text = re.sub(r'[`‘’‛⸂⸃⸌⸍⸜⸝]', "'", text)
	text = re.sub(r'[„“”]|(\'\')|(,,)', '"', text)
	return text

'''
This method was taken from Yeh et. al. (Semeval 2019 Task 4 Submission)
https://github.com/chialun-yeh/SemEval2019/blob/4cf5b57960100a41943cbba60d7413b0bab100fd/utils.py

Cleans the given text
'''
def clean_text(text):
	# remove URLs
	text = re.sub(r'(www\S+)|(https?\S+)|(href)', ' ', text)
	# remove anything within {} or [] or ().
	text = re.sub(r'\{[^}]*\}|\[[^]]*\]|\([^)]*\)', ' ', text)
	# remove irrelevant news usage
	text = re.sub(r'Getty [Ii]mages?|Getty|[Ff]ollow us on [Tt]witter|MORE:|ADVERTISEMENT|VIDEO', ' ', text)
	# remove @ or # tags or weird ......
	text = re.sub(r'@\S+|#\S+|\.{2,}', ' ', text)
	# remove newline in the beginning of the file
	text = text.lstrip().replace('\n','')
	# remove multiple white spaces
	re1 = re.compile(r'  +')
	text = re1.sub(' ', text)
	return _fix(text)

def tokenize(text):
	tokens = word_tokenize(text)
	return tokens

def lowercase(tokens_list):
	return list(map(lambda x: x.replace(x, x.lower()), tokens_list))

def remove_stopwords(tokens_list):

	stopwords_list = set(stopwords.words('english'))

	filtered_tokens = []
	for token in tokens_list:
		if token not in stopwords_list and (token.isalpha() or token in ALLOWED_PUNCTUATION):
			filtered_tokens.append(token)
	return filtered_tokens

'''
Removes stopwords BEFORE tokenizing. Necessary for the BERT classifier
'''
def remove_stopwords_untokenized(text):

	stopwords_list = set(stopwords.words('english'))
	text = ' '.join([word for word in text.split() if word not in stopwords_list])
	return text


'''
Preprocessing method for Doc2Vec classifiers
'''
def preprocess(text):
	text = clean_quotations(text)
	text = clean_text(text)
	tokenized = tokenize(text)
	lowercased = lowercase(tokenized)
	stopworded = remove_stopwords(lowercased)
	return stopworded

'''
Preprocessing method for the BERT classifier
(Tokenization happens inside of BERT since we need to add things
like [CLS] and [SEP] tokens)
'''
def bert_preprocess(text):
	text = clean_quotations(text)
	text = clean_text(text)
	text = remove_stopwords_untokenized(text)
	return text



if __name__ == '__main__':

	text = "Lorem Ipsum is simply dummy text of the printing and typesetting."
	'''
	tok = tokenize(text)
	print(tok)
	lower = lowercase(tok)
	print(lower)
	nostop = remove_stopwords(lower)
	print(nostop)
	'''

	print(preprocess(text))
