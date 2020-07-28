import torch
import nltk
import torchtext

from nltk.corpus import brown
from torchtext.data import Field

from model.LSTMTagger import LSTMTagger

DEVICE = torch.device('cpu')


nltk.download('brown')
nltk.download('universal_tagset')
brown_tagged_sents = brown.tagged_sents(tagset="universal")

data = [list(zip(*sent)) for sent in brown_tagged_sents]
WORD = Field(lower=True)
TAG = Field(unk_token=None)


examples = []
for words, tags in data:
    examples.append(
        torchtext.data.Example.fromlist(
            [list(words), list(tags)], fields=[('words', WORD), ('tags', TAG)]
            )
        )

dataset = torchtext.data.Dataset(examples, fields=[('words', WORD), ('tags', TAG)])

WORD.build_vocab(dataset.words, min_freq=0.2)
TAG.build_vocab(dataset.tags)

INPUT_DIM = len(WORD.vocab)
OUTPUT_DIM = len(TAG.vocab)
EMB_DIM = 200
HID_DIM = 50
DROPOUT = 0.5
BIDIRECTIONAL = True
MODEL = LSTMTagger(INPUT_DIM, EMB_DIM, HID_DIM,
                   OUTPUT_DIM, DROPOUT, BIDIRECTIONAL)
checkpoint = torch.load('resources/best-val-model.pt')
MODEL.load_state_dict(checkpoint)
