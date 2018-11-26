# Import HMM module
from nltk.tag import hmm

import random
from nltk.stem import PorterStemmer
from nltk.corpus import brown


SEED = 42
random.seed(SEED)

all_data = list(brown.tagged_sents(tagset='universal'))

random.shuffle(all_data)
i = int(len(all_data)*0.2)

porter = PorterStemmer()

"""
train_data = [[(porter.stem(word.lower()), tag) for word, tag in sent] for sent in all_data[:i]]
test_data = [[(porter.stem(word.lower()), tag) for word, tag in sent] for sent in all_data[i:]]

trainer = hmm.HiddenMarkovModelTrainer()
tagger = trainer.train_supervised(train_data)

print(tagger.evaluate(test_data))
"""

for _ in range(200):
    random.shuffle(all_data)
    train_data = [ [(porter.stem(word.lower()), tag) for word, tag in sent] for sent in all_data[:i]]
    test_data = [ [(porter.stem(word.lower()), tag) for word, tag in sent] for sent in all_data[i:]]
    trainer = hmm.HiddenMarkovModelTrainer()
    tagger = trainer.train_supervised(train_data)
    print(tagger.evaluate(train_data), tagger.evaluate(test_data))