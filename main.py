from evaluate import create_eval_pipeline, score, detailed_score, save_predictions
from read import read_articles_file, split_articles, reorder_labels_file
from doc2vec import train_doc2vec_model, extract_doc2vec_representations, enhance_doc2vec_representations
from pretrained_bert import bert_mask_input, get_bert_model, load_bert_model, make_predictions
from transformers import BertTokenizer
from utils import labels_to_integers
import numpy as np
import tensorflow_hub as hub
import tensorflow as tf

def doc2vec_main():

	articles_path = './datasets/articles-training-byarticle-20181122.xml'
	labels_path = './datasets/ground-truth-training-byarticle-20181122.xml'
	results, labels = read_articles_file(articles_path, labels_path)

	dev_dataset_size = 0.25
	train_labels, dev_labels, train, dev = split_articles(results, labels, dev_size=dev_dataset_size)


	test_articles_path = './datasets/articles-test-byarticle-20181207.xml'
	test_labels_path = './datasets/ground-truth-test-byarticle-20181207.xml'
	reorder_labels_file(test_labels_path)
	test, test_labels = read_articles_file(test_articles_path, test_labels_path)

	#
	train_vector_size=50
	train_epochs=100
	train_alpha=0.025
	# train_doc2vec_model(train, vector_size=train_vector_size, epochs=train_epochs, alpha=train_alpha)

	extract_epochs=100
	extract_alpha=0.025
	extract_doc2vec_representations(train, epochs=extract_epochs, alpha=extract_alpha)
	extract_doc2vec_representations(dev, epochs=extract_epochs, alpha=extract_alpha)
	extract_doc2vec_representations(test, epochs=extract_epochs, alpha=extract_alpha)

	print('Train dataset:')
	pipeline = create_eval_pipeline(train, train_labels)
	predictions, accuracy, precision, recall, f1 = detailed_score(pipeline, train, train_labels)

	print('Dev Dataset:')
	dev_pipeline = create_eval_pipeline(dev, dev_labels)
	dev_pred, dev_acc, dev_prec, dev_rec, dev_f1 = detailed_score(dev_pipeline, dev, dev_labels)

	print('Test Dataset:')
	test_pipeline = create_eval_pipeline(test, test_labels)
	test_pred, test_acc, test_prec, test_rec, test_f1 = detailed_score(test_pipeline, test, test_labels)

	print('Saving predictions in predictions folder...')
	save_predictions(predictions, 'predictions/doc2vec_train_predictions')
	save_predictions(dev_pred, 'predictions/doc2vec_dev_predictions')
	save_predictions(test_pred, 'predictions/doc2vec_test_predictions')

def doc2vec_enhanced_main():

	articles_path = './datasets/articles-training-byarticle-20181122.xml'
	labels_path = './datasets/ground-truth-training-byarticle-20181122.xml'
	results, labels = read_articles_file(articles_path, labels_path)

	dev_dataset_size = 0.25
	train_labels, dev_labels, train, dev = split_articles(results, labels, dev_size=dev_dataset_size)


	test_articles_path = './datasets/articles-test-byarticle-20181207.xml'
	test_labels_path = './datasets/ground-truth-test-byarticle-20181207.xml'
	reorder_labels_file(test_labels_path)
	test, test_labels = read_articles_file(test_articles_path, test_labels_path)

	train_vector_size=50
	train_epochs=100
	train_alpha=0.025
	#train_doc2vec_model(train, vector_size=train_vector_size, epochs=train_epochs, alpha=train_alpha)

	extract_epochs=100
	extract_alpha=0.025
	extract_doc2vec_representations(train, epochs=extract_epochs, alpha=extract_alpha)
	extract_doc2vec_representations(dev, epochs=extract_epochs, alpha=extract_alpha)
	extract_doc2vec_representations(test, epochs=extract_epochs, alpha=extract_alpha)

	# We enhance our representations with the handcreafted features
	enhance_doc2vec_representations(train)
	enhance_doc2vec_representations(dev)
	enhance_doc2vec_representations(test)

	print('Train dataset:')
	pipeline = create_eval_pipeline(train, train_labels)
	predictions, accuracy, precision, recall, f1 = detailed_score(pipeline, train, train_labels)

	print('Dev Dataset:')
	dev_pipeline = create_eval_pipeline(dev, dev_labels)
	dev_pred, dev_acc, dev_prec, dev_rec, dev_f1 = detailed_score(dev_pipeline, dev, dev_labels)

	print('Test Dataset:')
	test_pipeline = create_eval_pipeline(test, test_labels)
	test_pred, test_acc, test_prec, test_rec, test_f1 = detailed_score(test_pipeline, test, test_labels)

	print('Saving predictions in predictions folder...')
	save_predictions(predictions, 'predictions/enhanced_doc2vec_train_predictions')
	save_predictions(dev_pred, 'predictions/enhanced_doc2vec_dev_predictions')
	save_predictions(test_pred, 'predictions/enhanced_doc2vec_test_predictions')


def bert_main():

	articles_path = './datasets/articles-training-byarticle-20181122.xml'
	labels_path = './datasets/ground-truth-training-byarticle-20181122.xml'
	results, labels = read_articles_file(articles_path, labels_path)

	dev_dataset_size = 0.25
	train_labels, dev_labels, train, dev = split_articles(results, labels, dev_size=dev_dataset_size)


	test_articles_path = './datasets/articles-test-byarticle-20181207.xml'
	test_labels_path = './datasets/ground-truth-test-byarticle-20181207.xml'
	reorder_labels_file(test_labels_path)
	test, test_labels = read_articles_file(test_articles_path, test_labels_path)




	ml = 512
	alpha = 2e-05
	epsilon = 1e-08
	tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

	train_input, train_mask = bert_mask_input(train, ml, tokenizer, alpha, epsilon)
	dev_input, dev_mask = bert_mask_input(dev, ml, tokenizer, alpha, epsilon)
	test_input, test_mask = bert_mask_input(test, ml, tokenizer, alpha, epsilon)

	train_label = tf.convert_to_tensor(labels_to_integers(train_labels))
	dev_label = tf.convert_to_tensor(labels_to_integers(dev_labels))
	test_label = tf.convert_to_tensor(labels_to_integers(test_labels))


	model_save_path = './models/pretrained_bert_model.tf'
	callbacks = [tf.keras.callbacks.ModelCheckpoint(filepath=model_save_path, save_weights_only=True, monitors='val_loss',
													model='min', save_best_only=True)]

	epochs = 5
	model = get_bert_model(train_input, train_mask, train_label, test_input, test_mask, test_label, callbacks, model_save_path, num_epochs=epochs)
	loaded_model = load_bert_model('./models/pretrained_bert_model.tf')
	print(loaded_model.summary())


	#predictions = make_predictions(loaded_model, test_input, test_mask, callbacks)
	#print(predictions)

	#loss, accuracy = loaded_model.evaluate([train_input, train_mask], train_label, verbose=2)
	#print("Untrained model, accuracy: {:5.2f}%".format(100 * acc))

	#train_predictions = loaded_model.evaluate(train)
	#print(train_predictions)

if __name__ == '__main__':

	doc2vec_main()
	doc2vec_enhanced_main()
	#bert_main()
