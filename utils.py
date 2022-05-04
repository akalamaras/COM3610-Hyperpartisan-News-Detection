from xml import sax
import pandas as pd
import csv

ALLOWED_PUNCTUATION = '.?!,'

class GroundTruthHandler(sax.ContentHandler):

	def __init__(self, label):
		sax.ContentHandler.__init__(self)
		self.label = label

	def startElement(self, name, attrs):
		if name == "article":
			self.label.append(attrs.getValue("hyperpartisan"))

def list_to_dataframe(article_list):

	df = pd.DataFrame(columns=['text', 'label'])
	for article in article_list:
		df_new_row = pd.DataFrame({'text': [article.bert_preprocessed],
				   	   			   'label': [article.is_hyperpartisan_int]
				   	   			  })
		df = pd.concat([df, df_new_row])
	return df

def labels_to_integers(labels_list):

	result = []
	for label in labels_list:
		integer_label = 1 if label == 'true' else 0
		result.append(integer_label)
	return result

'''
def write_to_csv(article_list, save_path):

	header = ['title', 'text', 'label']

	data = []
	for article in article_list:
		data_entry = [article.plain_title, article.plain_text, article.is_hyperpartisan_int]
		data.append(data_entry)

	with open(save_path, 'w', encoding='UTF8', newline='') as f:

		writer = csv.writer(f)

		writer.writerow(header)
		writer.writerows(data)
'''
