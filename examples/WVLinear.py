import numpy
import time
import sys
import subprocess
import os
import random
import theano
import numpy as np
from sklearn.neural_network import MLPClassifier

from is13.rnn import elman_attention
from is13.data import load
from is13.metrics.accuracy import conlleval
from is13.utils.tools import *
from sklearn.metrics import *


def getX(input, wv, iw, wi):
    x = []
    for wordList in input:
        v = []
        for word in wordList:
            i = 0
            if word in wi:
                i = wi[word]
            v.append(wv[i])
        v = np.array(v).flatten()
        x.append(v)
    return x


def getInputOutput(lex, y, win, idx2word):
    input = []
    output = []
    for i in range(len(lex)):
        wordListList = original_contextwin(lex[i], win)
        for j in range(len(wordListList)):
            wordList = wordListList[j]
            realWordList = [idx2word[word] for word in wordList]
            input.append(realWordList)
            output.append(y[i][j])
    return input, output


def getFormatedPY(y, py, idx2label):
    fpy = []
    index = 0
    for i in range(len(y)):
        v = []
        for j in range(len(y[i])):
            v.append(idx2label[py[index]])
            index += 1
        fpy.append(v)
    return fpy

def test(s):

    # load word vector
    wv = np.load("./../WV/" + s['WVFolderName'] + "/" + s['model']+".words" + str(s['emb_dimension']) + ".npy")
    # load vocab
    with open("./../WV/" + s['WVFolderName'] + "/" + s['model']+".words" + str(s['emb_dimension']) + ".vocab") as f:
        vocab = [line.strip() for line in f if len(line) > 0]
    wi = dict([(a, i) for i, a in enumerate(vocab)])
    iw = vocab

    # load the dataset
    if s['dataset'] == 'atis':
        train_set, valid_set, test_set, dic = load.atisfold(s['fold'])
    if s['dataset'] == 'ner':
        train_set, valid_set, test_set, dic = load.ner()
    if s['dataset'] == 'chunk':
        train_set, valid_set, test_set, dic = load.chunk()
    if s['dataset'] == 'pos':
        train_set, valid_set, test_set, dic = load.pos()

    idx2label = dict((k, v) for v, k in dic['labels2idx'].iteritems())
    idx2word = dict((k, v) for v, k in dic['words2idx'].iteritems())

    train_lex, train_ne, train_y = train_set
    valid_lex, valid_ne, valid_y = valid_set
    test_lex, test_ne, test_y = test_set

    # train_lex.extend(valid_lex)
    # train_ne.extend(valid_ne)
    # train_y.extend(valid_y)

    vocsize = len(dic['words2idx'])
    nclasses = len(dic['labels2idx'])

    my_train_input, my_train_y = getInputOutput(train_lex, train_y, s['win'], idx2word)
    my_train_x = getX(my_train_input, wv, iw, wi)

    my_test_input, my_test_y = getInputOutput(test_lex, test_y, s['win'], idx2word)
    my_test_x = getX(my_test_input, wv, iw, wi)

    clf = MLPClassifier(hidden_layer_sizes=(), verbose=False, activation='tanh')
    clf.fit(my_train_x, my_train_y)



    # eval
    eval_options = []
    if s['dataset'] == 'pos':
        eval_options = ['-r']
    my_train_yp = clf.predict(my_train_x)
    my_test_yp = clf.predict(my_test_x)
    # print my_train_y
    # print my_train_yp
    predictions_train = getFormatedPY(train_y, my_train_yp, idx2label)
    groundtruth_train = [map(lambda x: idx2label[x], y) for y in train_y]
    words_train = [map(lambda x: idx2word[x], w) for w in train_lex]

    predictions_test = getFormatedPY(test_y, my_test_yp, idx2label)
    groundtruth_test = [map(lambda x: idx2label[x], y) for y in test_y]
    words_test = [map(lambda x: idx2word[x], w) for w in test_lex]

    res_train = conlleval(predictions_train, groundtruth_train, words_train, folder + '/linear.train.' + s['dataset'] + '.txt', eval_options)
    res_test = conlleval(predictions_test, groundtruth_test, words_test, folder + '/linear.test.' + s['dataset'] + '.txt', eval_options)

    # print '                        train', res_train['p'], res_train['r'], res_train['f1'] , ' ' * 20
    # print '                         test', res_test['p'], res_test['r'], res_test['f1'] , ' ' * 20
    print res_test['f1'],

if __name__ == '__main__':
    folder = os.path.basename(__file__).split('.')[0]
    if not os.path.exists(folder): os.mkdir(folder)
    print folder
    for model in ['glove', 'skip', 'cbow']: #
        for emb_dimension in [25, 50, 100, 250, 500]:
            for WVFolderName in ['201308_p_lr_linear-2_' , '201308_p_lr_dependency-1_'] :
                    #['201308_p_word_linear-2_' , '201308_p_structured_linear-2_' , '201308_p_word_dependency-1_', '201308_p_structured_dependency-1_'] :

                print (WVFolderName + model), "\t", emb_dimension, "\t",
                for dataset in ['pos','chunk', 'ner', 'atis']:
                    s = {
                        'verbose': 2,
                        'fold': 3,  # 5 folds 0,1,2,3,4
                        # 'dataset' : 'atis',
                        'win': 5,
                        'seed': 123,
                    }
                    s['WVFolderName'] = WVFolderName + model
                    s['model'] = 'sgns'
                    if model == 'glove':
                        s['WVFolderName'] = WVFolderName + 'skip'
                        s['model'] = 'glove'
                    s['emb_dimension'] = emb_dimension
                    s['dataset'] = dataset


                    test(s)
                    pass
                print ''
