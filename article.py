from preprocessing import preprocess
import features
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

		# Doc2Vec

		# Used to train the Doc2Vec model (training data only)
		self.doc2vec_docs = [TaggedDocument(sentence, [i]) for i, sentence in enumerate(self.sentences)]
		#  Doc2Vec representations for the article text, set using  extract_doc2vec_representations
		#  when the Doc2Vec model is trained
		self.doc2vec_representation = None


def to_evaluation_format(article_list):
	return [article.doc2vec_representation for article in article_list]
