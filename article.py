from preprocessing import preprocess, bert_preprocess
import features
import numpy as np
from gensim.models.doc2vec import TaggedDocument

class Article(object):

	def __init__(self, title, text, label):

		self.plain_title = title
		self.plain_text = text
		self.is_hyperpartisan = label

		self.title = preprocess(title)
		self.text = preprocess(text)
		self.sentences = features.get_sentences(self.text)

		# Features
		self.word_count = features.get_word_count(self.text)
		self.char_count = features.get_char_count(self.text)
		self.average_word_length = features.get_average_word_length(self.text)
		self.sentence_count = features.get_sentence_count(self.text)
		self.average_sentence_length = features.get_average_sentence_length(self.text)
		self.biased_word_count = features.get_biased_word_count(self.text)

		# Doc2Vec

		# Used to train the Doc2Vec model (training data only)
		self.doc2vec_docs = [TaggedDocument(sentence, [i]) for i, sentence in enumerate(self.sentences)]
		#  Doc2Vec representations for the article text, set using  extract_doc2vec_representations
		#  when the Doc2Vec model is trained
		self.doc2vec_representation = None

		# BERT

		# The BERT model requires its own preprocessing (for [CLS], [SEP], [PAD] tokens etc.)
		self.bert_preprocessed = bert_preprocess(title) + bert_preprocess(text)
		self.is_hyperpartisan_int = 1 if label == 'true' else 0


def to_evaluation_format(article_list):
	return [article.doc2vec_representation for article in article_list]


def get_handpicked_features(article):

	text_features = [article.word_count,
					 article.char_count,
					 article.average_word_length,
					 article.sentence_count,
					 article.average_sentence_length,
					 article.biased_word_count]
	return np.asarray(text_features)
