from utils import list_to_dataframe, labels_to_integers
from preprocessing import clean_quotations, clean_text
from transformers import BertTokenizer, TFBertForSequenceClassification
from read import read_articles_file, split_articles
import tensorflow as tf
from tensorflow.keras import backend as b


'''
Creates the input_ids and attention_masks to input into the pre-trained BERT classifier
Parameters:
    article_list: List of Article objects
	max_seq_length: max sequence length
	tokenizer: the BertTokenizer
	alpha: the learning rate
	epsilon: the numerical stability of the optimizer
Returns:
    the input_ids tensor
	the attention_masks tensor
'''
def bert_mask_input(article_list, max_seq_length, tokenizer, alpha=2e-5, epsilon=1e-08):

	input_ids = []
	attention_masks = []

	for article in article_list:
		encoded_dict = tokenizer.encode_plus(article.bert_preprocessed,
											add_special_tokens=True,
											max_length=max_seq_length,
											pad_to_max_length=True,
											return_attention_mask=True)
		input_ids.append(encoded_dict['input_ids'])
		attention_masks.append(encoded_dict['attention_mask'])

	return tf.convert_to_tensor(input_ids), tf.convert_to_tensor(attention_masks)

'''
Loads and fine-tunes the BERT model.
Parameters:
   train_input: The input_ids tensor for the training set
   train_mask: The attention_masks tensor for the training set
   train_label: The tensor containing the labels for the training set
   dev_input: The input_ids tensor for the dev set
   dev_mask: The attention_masks tensor for the dev set
   dev_label: The tensor containing the labels for the dev set
   callbacks: A list of callbacks
   model_save_path: The path where the model will be saved
   alpha: The learning rate
   epsilon: The numerical stability of the optimizer
   num_epochs: The number of training epochs
'''
def get_bert_model(train_input, train_mask, train_label, dev_input, dev_mask, dev_label, callbacks, model_save_path, alpha = 2e-05, epsilon=1e-08, num_epochs=4):

	bert_model = TFBertForSequenceClassification.from_pretrained('bert-base-cased', num_labels=True)

	loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
	acc_metric = tf.keras.metrics.BinaryAccuracy('binary_accuracy')

	optimizer = tf.keras.optimizers.Adam(learning_rate=alpha, epsilon=epsilon)

	bert_model.compile(loss=loss, metrics=[acc_metric], optimizer=optimizer)
	history = bert_model.fit([train_input, train_mask],
							  train_label,
							  batch_size=4,
							  epochs=num_epochs,
							  validation_data=([dev_input, dev_mask], dev_label),
							  shuffle=True,
							  callbacks=callbacks)

	print('Saving BERT model to path: {}'.format(model_save_path))
	bert_model.save(model_save_path, save_format="tf")

'''
Loads the BERT model from the specified path
Parameters:
    model_save_path: The path where the BERT model we want to load is saved
'''
def load_bert_model(model_save_path):

	model = tf.keras.models.load_model(model_save_path)
	return model

def make_predictions(bert_model, predict_input, predict_mask, callbacks):

	pred = bert_model.predict([predict_input, predict_mask],
							   batch_size=4,
							   callbacks=callbacks)
	return pred
