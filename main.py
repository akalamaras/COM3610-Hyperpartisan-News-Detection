from evaluate import create_eval_pipeline, score, detailed_score, save_predictions
from read import read_articles_file, split_articles
from doc2vec import train_doc2vec_model, extract_doc2vec_representations

def doc2vec_main():

	articles_path = './datasets/articles-training-byarticle-20181122.xml'
	labels_path = './datasets/ground-truth-training-byarticle-20181122.xml'
	results, labels = read_articles_file(articles_path, labels_path)
	train_labels, test_labels, train, test = split_articles(results, labels, dev_size=0.25)

	extract_epochs=100

	# train_doc2vec_model(train, vector_size=200)
	extract_doc2vec_representations(train, epochs=extract_epochs)
	extract_doc2vec_representations(test, epochs=extract_epochs)

	print('Train')
	pipeline = create_eval_pipeline(train, train_labels)
	predictions, accuracy, precision, recall, f1 = detailed_score(pipeline, train, train_labels)
	print('-------------------------------------------')

	print('Test')
	pipeline2 = create_eval_pipeline(test, test_labels)
	predictions2, accuracy2, precision2, recall2, f12 = detailed_score(pipeline2, test, test_labels)

	save_predictions(predictions, 'predictions/train_predictions')
	save_predictions(predictions2, 'predictions/test_predictions')


if __name__ == '__main__':

	doc2vec_main()
