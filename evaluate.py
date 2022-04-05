from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from article import to_evaluation_format
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
import numpy as np

'''
Creates an evaluation pipeline using the training data
to fit it
Parameters:
	train_articles: The training articles list
	train_labels: The corresponding labels list
Returns:
	pipeline, an sklearn Pipeline object fit to the training data
'''
def create_eval_pipeline(train_articles, train_labels):

	eval_format = to_evaluation_format(train_articles)

	pipeline = Pipeline([('scaler', StandardScaler()), ('svc', SVC())])
	pipeline.fit(eval_format, train_labels)
	return pipeline

'''
Scores a list of articles on the specified pipe lineby predicting its labels
and comparing them to the supplied label list
Parameters:
	pipeline: The Pipeline object (correctly fit)
	articles_list: The list of articles to be scored
	labels_list: The correct labels (to which the predictions are compared)
Returns:
	the basic score (accuracy)
'''
def score(pipeline, articles_list, labels_list):

	eval_list = to_evaluation_format(articles_list)
	s = pipeline.score(eval_list, labels_list)
	print('Basic Score: {}'.format(s))
	return s

'''
Scores a list of articles on the specified pipe lineby predicting its labels
and comparing them to the supplied label list. More detailed than score
Parameters:
	pipeline: The Pipeline object (correctly fit)
	articles_list: The list of articles to be scored
	labels_list: The correct labels (to which the predictions are compared)
Returns:
	predictions: the predicted labels list
	accuracy:
	recall:
	precision:
	f1-score:
'''
def detailed_score(pipeline, articles_list, labels_list):

	print('Calculated scores...')

	eval_list = to_evaluation_format(articles_list)
	predictions = pipeline.predict(eval_list)

	accuracy = accuracy_score(labels_list, predictions)
	precision = precision_score(labels_list, predictions, pos_label='true')
	recall = recall_score(labels_list, predictions, pos_label='true')
	f1_score = 2 * (precision * recall) / (precision + recall)

	print('Detailed Scores: ')
	print('accuracy: {}'.format(accuracy))
	print('precision: {}'.format(precision))
	print('recall: {}'.format(recall))
	print('f1-score: {}'.format(f1_score))

	return predictions, accuracy, precision, recall, f1_score


'''
Saves the predictions to the specified path
Parameters:
	predictions_list: List of predictions
	path: The relative path
'''
def save_predictions(predictions_list, path):

	print('Saving predictions to path: {}...'.format(path))
	np.save(path, predictions_list)
