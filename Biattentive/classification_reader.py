from allennlp.data import DatasetReader, Instance
from allennlp.data.fields import LabelField, TextField
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.tokenizers import WordTokenizer
from allennlp.common.file_utils import cached_path

# Data will be formatted as:
# in our case:
# [text][tab][label]

@DatasetReader.register('classification-tsv')
class ClassificationTsvReader(DatasetReader):
    def __init__(self):
        self.tokenizer = WordTokenizer()
        self.token_indexers = {'tokens': SingleIdTokenIndexer()}
        self._cache_directory = None

    def _read(self, file_path: str):
        with open(file_path, 'r') as lines:
            for line in lines:
                text, label = line.strip().split('\t')
                text_field = TextField(self.tokenizer.tokenize(text),
                                       self.token_indexers)
                label_field = LabelField(label)
                fields = {'text': text_field,
                          'label': label_field}
                yield Instance(fields)

#reader = ClassificationTsvReader()
#print(reader.read(cached_path('example_test.tsv')))
