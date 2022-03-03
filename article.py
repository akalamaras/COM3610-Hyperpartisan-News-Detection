from utils import clean_text, clean_quotations
from lxml.etree import iterparse
from preprocessing import tokenize

class Article(object):

	def __init__(self, title, text):
		self.title = clean_quotations(title)
		self.text = clean_text(clean_quotations(text))

	def __str__(self):
		return self.title + '.' + self.text


def read_articles_file(articles):

	results = []

	for event, elem in iterparse(articles):
		if elem.tag == 'article':

			# Id not necessary so scrap
			# id = elem.attrib['id']
			title = elem.attrib['title']
			text = "".join(elem.itertext())
			results.append(Article(title, text))
			elem.clear()
	return results

if __name__ == '__main__':

	path = './datasets/articles-training-byarticle-20181122.xml'
	res = read_articles_file(path)
	ar1 = res[0]
	print(tokenize(ar1.text))
