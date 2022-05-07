from xml import sax
from utils import GroundTruthHandler
from lxml.etree import iterparse
from sklearn.model_selection import train_test_split
from article import Article
import xml.etree.ElementTree as ET

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

def read_labels_file(labels_path):

	article_labels = []
	with open(labels_path) as l:
		sax.parse(l, GroundTruthHandler(article_labels))
	return article_labels

'''
Splits a list of Article objects into 2 sets (train & test) so that the label
percentages are similar
Parameters:
	articles: A list of Article objects
	labels: A list of booleans indicating labels (indices correspond to the articles list)
	dev_size: The size of the dev dataset (default: 0.5)
Returns:
	train_articles: The training dataset of Article objects
	dev_articles: The dev dataset of Article objects
'''
def split_articles(articles, labels, dev_size = 0.5):

	print('Splitting articles into train-dev sets...')

	train_labels, dev_labels, train_articles, dev_articles = train_test_split(labels, articles, test_size=dev_size, shuffle=True)
	return train_labels, dev_labels, train_articles, dev_articles

'''
Reorders the test dataset labels based on ID.
This is necessary due to how the test labels were provided
'''
def reorder_labels_file(labels_path):
	print('Reordering the Test Datasets labels based on id...')

	tree = ET.parse('./datasets/ground-truth-test-byarticle-20181207.xml')
	articles = tree.getroot()

	elements = articles.findall("*[@id]")
	new_elements = sorted(elements, key=lambda article: (article.tag, article.attrib['id']))
	articles[:] = new_elements

	tree.write('./datasets/ground-truth-test-byarticle-20181207.xml', xml_declaration=True, encoding='utf-8')

if __name__ == '__main__':


	test_articles_path = './datasets/articles-test-byarticle-20181207.xml'
	test_labels_path = './datasets/ground-truth-test-byarticle-20181207.xml'
	reorder_labels_file(test_labels_path)
	test, test_labels = read_articles_file(test_articles_path, test_labels_path)



	print(test[1].sentences)
	print('----------------------------------------------')
	print(test[1].is_hyperpartisan)
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
