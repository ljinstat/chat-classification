from classification_reader import ClassificationTsvReader
import torch
import torch.optim as optim
from allennlp.common.file_utils import cached_path
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.text_field_embedders import TextFieldEmbedder, BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.models import BiattentiveClassificationNetwork
from allennlp.modules.seq2vec_encoders import PytorchSeq2VecWrapper
from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper
from allennlp.predictors import TextClassifierPredictor
from allennlp.modules import FeedForward


reader = ClassificationTsvReader()
train_path = 'your_path/file'
validation_path = 'your_path/file'
train_data = reader.read(cached_path('train_path')) # add tsv file here
validation_data = reader.read(cached_path('validation_path')) # add tsv file here
# vocabulary the mapping[s] from tokens / labels to ids
vocab = Vocabulary.from_instances(train_data + validation_data)

# BasicTextFieldEmbedder which takes a mapping from index names to embeddings.
# It's also possible to start with pre-trained embeddings (for example, GloVe vectors)
EMBEDDING_DIM = 10
HIDDEN_DIM = 10
token_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokeds'), \
                            embedding_dim=EMBEDDING_DIM)
word_embeddings = BasicTextFieldEmbedder({"tokens": token_embedding})

# To classify each sentence, we need to convert the sequence of embeddings into a single vector.
# In AllenNLP, the model that handles this is referred to as a Seq2VecEncoder:
# a mapping from sequences to a single vector.
encoder = PytorchSeq2VecWrapper(torch.nn.LSTM(EMBEDDING_DIM, HIDDEN_DIM, bidirectional=True, batch_first=True))
feedforward = FeedForward(input_dim=EMBEDDING_DIM, num_layers=1, hidden_dims=[256], activations=[torch.nn.ReLU])
integrator_encoder = PytorchSeq2SeqWrapper(torch.nn.LSTM(EMBEDDING_DIM, HIDDEN_DIM, batch_first=True))
# model: parameters have to be modified according to the paper
model = BiattentiveClassificationNetwork(text_field_embedder=word_embeddings,
                                         encoder=encoder,
                                         vocab=vocab,
                                         embedding_dropout=0.0,
                                         pre_encode_feedforward=feedforward,
                                         integrator=integrator_encoder,
                                         integrator_dropout=0.0,
                                         output_layer=feedforward,
                                         elmo=None) # not sure about the parameters

# check if we get access to a GPU
if torch.cuda.is_available():
    cuda_device = 0
    model = model.cuda(cuda_device)
else:
    cuda_device = -1

# optimazation
optimizer = optim.SGD(model.parameters(), lr=0.001) # optimizer depends on the original paper

# we need a DataIterator that handles batching for our datasets, sort sentences with similar lengths together
iterator = BucketIterator(batch_size=2, sorting_keys=[('sentence', 'num_tokens')]) # BasicIterator(batch_size=2)
iterator.index_with(vocab)

# Training
trainer = Trainer(model=model,
                  optimizer=optimizer,
                  iterator=iterator,
                  train_dataset=train_data,
                  validation_dataset=validation_data,
                  patience=2, # early stopping if it is stuck for 2 epochs
                  num_epochs=10, # we should increase the number of epoch later, it's just for early try
                  cuda_device=cuda_device)
trainer.train()

# predictor
pre_example = 'Good morning'
predictor = TextClassifierPredictor()
pre = predictor.predict(pre_example)

# save the model
with open('./tmp/classifier_biattention_model.th', 'wb') as f:
    torch.save(model.state_dict(), f)

# save the vocabulary
vocab.save_to_files('./tmp/vocabulary')

# reload the model
# vocab2 = Vocabulary.from_files('./tmp/vocabulary')
# model2 = BiattentiveClassificationNetwork(word_embeddings, encoder, vocab2)
# with open('./tmp/classifier_biattention_model.th', 'rb') as f:
#     model2.load_state_dict(torch.load(f))
# if cuda_device > -1:
#     model2.cuda(cuda_device)
# predictor2 = TextClassifierPredictor()
