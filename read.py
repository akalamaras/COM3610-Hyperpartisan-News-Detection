from xml import sax
from utils import GroundTruthHandler
from lxml.etree import iterparse
from sklearn.model_selection import train_test_split
from article import Article

'''
Reads the dataset containing the articles , as well as the one with the
corresponding labels and create
Parameters:
	articles_path: The path to the file containing the articles
	labels_path: The path to the file containing the labels for the articles (ground truth)
Returns:
	results: A list of Article objects (1-1 correspondence to the articles in articles_path)
	article_labels: A list of booleans indicating labels (indices correspond to the results list)
'''
def read_articles_file(articles_path, labels_path):

	print('Reading articles file at path: {}...'.format(articles_path))

	article_labels = []
	with open(labels_path) as l:
		sax.parse(l, GroundTruthHandler(article_labels))

	results = []

	# labels index
	i = 0
	for event, elem in iterparse(articles_path):
		if elem.tag == 'article':

			# Id not necessary so scrap
			# id = elem.attrib['id']
			title = elem.attrib['title']
			text = "".join(elem.itertext())
			results.append(Article(title, text, article_labels[i]))
			i += 1
			elem.clear()

	# Despite the labels being part of the Article objcet in results, we return the
	# labels list anyway to make splitting the dataset into train-test easier
	return results, article_labels

'''
Splits a list of Article objects into 2 sets (train & test) so that the label
percentages are similar
Parameters:
	articles: A list of Article objects
	labels: A list of booleans indicating labels (indices correspond to the articles list)
	test_size: The size of the test dataset (default: 0.5)
Returns:
	train_articles: The training dataset of Article objects
	test_articles: The test dataset of Article objects
'''
def split_articles(articles, labels, dev_size = 0.5):

	print('Splitting articles into train-test sets...')

	train_labels, test_labels, train_articles, test_articles = train_test_split(labels, articles, test_size=dev_size, shuffle=True)
	return train_labels, test_labels, train_articles, test_articles


if __name__ == '__main__':

	articles_path = './datasets/articles-training-byarticle-20181122.xml'
	labels_path = './datasets/ground-truth-training-byarticle-20181122.xml'
	results, labels = read_articles_file(articles_path, labels_path)
	train_labels, test_labels, train, test = split_articles(results, labels)


	print(train[0].sentences)
	print('----------------------------------------------')
	print(train[0].doc2vec_docs)
	'''
	for article in train:
		print(article.title)
		print(article.word_count)
		print(article.char_count)
		print(article.average_word_length)
		print(article.sentence_count)
		print(article.average_sentence_length)
		print('------------------------------')
	'''
