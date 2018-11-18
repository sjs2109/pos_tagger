
# Import the toolkit and tags
import nltk
from nltk.corpus import treebank

# Train data - pretagged
train_data = treebank.tagged_sents()[:3000]

print(train_data[0])

# Import HMM module
from nltk.tag import hmm

# Setup a trainer with default(None) values
# And train with the data
trainer = hmm.HiddenMarkovModelTrainer()
tagger = trainer.train_supervised(train_data)

print(tagger)
# Prints the basic data about the tagger

print(tagger.tag("Today is a good day .".split()))

print(tagger.tag("Joe met Joanne in Delhi .".split()))

print(tagger.tag("Chicago is the birthplace of Ginny".split()))