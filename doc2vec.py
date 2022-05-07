from gensim.models.doc2vec import Doc2Vec
from article import Article, get_handpicked_features
import numpy as np



'''
Trains a Doc2Vec model for the training dataset of Articles and saves it in the
specified path
Parameters:
	train_articles: The training dataset of Article objects
	vector_size: The dimensionality of the feature vectors (default: 5)
	model_save_path: The relative path to save the Doc2Vec model (default: ./models/doc2vec_model)
'''
def train_doc2vec_model(train_articles, vector_size = 100, epochs=100, alpha=0.025, model_save_path='./models/doc2vec_model'):

	print('Training Doc2Vec model...')

	doc2vec_docs_2d = [article.doc2vec_docs for article in train_articles]
	# Model expects 1D list, so we flatten our 2D list with this oneliner
	doc2vec_docs_flat = [doc for article_doc in doc2vec_docs_2d for doc in article_doc]
	model = Doc2Vec(doc2vec_docs_flat, vector_size=vector_size, window=2, min_count=1, workers=4, epochs=epochs, alpha=alpha)
	model.save(model_save_path)

'''
Extracts the Doc2Vec representation for each article in the article list, according
to the specified trained Doc2Vec model
Parameters:
	articles_list: The list of articles
	model_path: The relative path to the trained Doc2Vec model
'''
def extract_doc2vec_representations(articles_list, epochs=100, alpha=0.025, model_path='./models/doc2vec_model'):

	print('Extracting Doc2Vec representations...')

	model = Doc2Vec.load(model_path)
	for article in articles_list:
		article.doc2vec_representation = model.infer_vector(article.text, epochs=epochs, alpha=alpha)

'''
Enhances the Doc2Vec model with handpicked features
Parameters:
	articles_list: The list of articles
'''
def enhance_doc2vec_representations(articles_list):

	for article in articles_list:
		features_to_add = get_handpicked_features(article)
		enhanced_doc2vec = np.hstack((article.doc2vec_representation, features_to_add))
		article.doc2vec_representation = enhanced_doc2vec
