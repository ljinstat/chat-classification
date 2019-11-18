from simpletransformers.classification import ClassificationModel
import pandas as pd
from sklearn.metrics import accuracy_score
import torch

# Train and Evaluation data needs to be in a Pandas Dataframe of two columns. The first column is the text with type str, and the second column is the label with type int.
train_file = 'your_training_data_file.tsv'# '../allennlp_cls/example_train.tsv'
train_df = pd.read_csv(train_file, sep='\t', header=0)

eval_file = 'your_training_data_file.tsv'# '../allennlp_cls/example_validation.tsv'
eval_df = pd.read_csv(eval_file, sep='\t', header=0)

# Create a ClassificationModel
model = ClassificationModel('xlnet', 'xlnet-base-cased', use_cuda=False) # You can set class weights by using the optional weight argument

# Train the model
model.train_model(train_df)

# Evaluate the model
result, model_outputs, wrong_predictions = model.eval_model(eval_df, acc=accuracy_score)
print('evaluation scores of the model {}'.format(result))

predictions, raw_outputs = model.predict(['your sentence here']) # please put your prediction here
print(predictions)
