
from keras.utils import plot_model
from keras.models import Sequential
from keras.layers import Dense, LSTM, InputLayer, Bidirectional, TimeDistributed, Embedding, Activation
from keras.optimizers import Adam
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras import backend as K
import numpy as np
from nltk.corpus import brown
from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(patience=3)
import matplotlib.pyplot as plt



def to_categorical(sequences, categories):
    cat_sequences = []
    for s in sequences:
        cats = []
        for item in s:
            cats.append(np.zeros(categories))
            cats[-1][item] = 1.0
        cat_sequences.append(cats)
    return np.array(cat_sequences)


def logits_to_tokens(sequences, index):
    token_sequences = []
    for categorical_sequence in sequences:
        token_sequence = []
        for categorical in categorical_sequence:
            token_sequence.append(index[np.argmax(categorical)])

        token_sequences.append(token_sequence)

    return token_sequences


def ignore_class_accuracy(to_ignore=0):
    def ignore_accuracy(y_true, y_pred):
        y_true_class = K.argmax(y_true, axis=-1)
        y_pred_class = K.argmax(y_pred, axis=-1)

        ignore_mask = K.cast(K.not_equal(y_pred_class, to_ignore), 'int32')
        matches = K.cast(K.equal(y_true_class, y_pred_class), 'int32') * ignore_mask
        accuracy = K.sum(matches) / K.maximum(K.sum(ignore_mask), 1)
        return accuracy

    return ignore_accuracy


if __name__ == '__main__':

    tagged_sentences = brown.tagged_sents(tagset='universal')
    sentences, sentence_tags = [], []
    for tagged_sentence in tagged_sentences:
        sentence, tags = zip(*tagged_sentence)
        sentences.append(np.array(sentence))
        sentence_tags.append(np.array(tags))

    (train_sentences,
     test_sentences,
     train_tags,
     test_tags) = train_test_split(sentences, sentence_tags, test_size=0.2)

    words, tags = set([]), set([])

    for s in train_sentences:
        for w in s:
            words.add(w.lower())

    for ts in train_tags:
        for t in ts:
            tags.add(t)

    word2index = {w: i + 2 for i, w in enumerate(list(words))}
    word2index['-PAD-'] = 0  # The special value used for padding
    word2index['-OOV-'] = 1  # The special value used for OOVs

    tag2index = {t: i + 1 for i, t in enumerate(list(tags))}
    tag2index['-PAD-'] = 0  # The special value used to padding

    train_sentences_X, test_sentences_X, train_tags_y, test_tags_y = [], [], [], []

    for s in train_sentences:
        s_int = []
        for w in s:
            try:
                s_int.append(word2index[w.lower()])
            except KeyError:
                s_int.append(word2index['-OOV-'])

        train_sentences_X.append(s_int)

    for s in test_sentences:
        s_int = []
        for w in s:
            try:
                s_int.append(word2index[w.lower()])
            except KeyError:
                s_int.append(word2index['-OOV-'])

        test_sentences_X.append(s_int)

    for s in train_tags:
        train_tags_y.append([tag2index[t] for t in s])

    for s in test_tags:
        test_tags_y.append([tag2index[t] for t in s])

    MAX_LENGTH = len(max(train_sentences_X, key=len))

    train_sentences_X = pad_sequences(train_sentences_X, maxlen=MAX_LENGTH, padding='post')
    test_sentences_X = pad_sequences(test_sentences_X, maxlen=MAX_LENGTH, padding='post')
    train_tags_y = pad_sequences(train_tags_y, maxlen=MAX_LENGTH, padding='post')
    test_tags_y = pad_sequences(test_tags_y, maxlen=MAX_LENGTH, padding='post')

    cat_train_tags_y = to_categorical(train_tags_y, len(tag2index))
    model = Sequential()
    model.add(InputLayer(input_shape=(MAX_LENGTH,)))
    model.add(Embedding(len(word2index), 128))
    model.add(Dense(len(tag2index)))
    model.add(Dense(len(tag2index)))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(0.001),
                  metrics=['accuracy', ignore_class_accuracy(0)])

    model.summary()

    hist = model.fit(train_sentences_X, to_categorical(train_tags_y, len(tag2index)), batch_size=2048, epochs=3, validation_split=0.2,callbacks=[early_stopping])

    score = model.evaluate(test_sentences_X, to_categorical(test_tags_y, len(tag2index)), verbose=0)
    print(model.metrics_names)

    print('model loss: {} acc: {}  ignore_accuracy : {}'.format(score[0], score[1],score[2]))
    plot_model(model, to_file='tmp/dnn_model_structure.png', show_shapes=True)

    fig, loss_ax = plt.subplots()

    acc_ax = loss_ax.twinx()

    loss_ax.plot(hist.history['loss'], 'y', label='train loss')
    loss_ax.plot(hist.history['val_loss'], 'r', label='val loss')

    acc_ax.plot(hist.history['acc'], 'b', label='train acc')
    acc_ax.plot(hist.history['val_acc'], 'g', label='val acc')

    loss_ax.set_xlabel('epoch')
    loss_ax.set_ylabel('loss')
    acc_ax.set_ylabel('accuray')

    loss_ax.legend(loc='upper left')
    acc_ax.legend(loc='lower left')
    plt.savefig("tmp/dnn_tagger_hist.png",dpi=300)

