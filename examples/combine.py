import numpy
import time
import sys
import subprocess
import os
import random
import pprint, pickle

from is13.rnn import elman, elman_moreH, elman_lstm, elman_attention, elman_attention_old
from is13.data import load
from is13.metrics.accuracy import conlleval
from is13.utils.tools import shuffle, minibatch, contextwin

if __name__ == '__main__':


    folder = os.path.basename(__file__).split('.')[0]
    if not os.path.exists(folder): os.mkdir(folder)



    filePath1 = './results/lr_single_n100drop02/'
    filePath2 = './results/lr_general_n100drop02_win33/'


    id = "concat_25"


    print filePath1
    print filePath2
    print id

    best = [[0,0] for i in xrange(11)]

    for ii in xrange(0, 25) :
        for jj in xrange(0, 25) :



            #i = 11
            #j = 15
           # if ii != jj:
              #  continue


            file1 = filePath1 + "s" + str(ii) + '.pkl'
            file2 = filePath2 + "s" + str(jj) + '.pkl'

            if os.path.isfile(file1) == False:
                continue
            if os.path.isfile(file2) == False:
                continue

            print 'epochs: ' , ii, '   ', jj

            output = open(file1, 'rb')
            [s1, s_train1, s_test1, s_valid1] = pickle.load(output)
            output.close()
            output = open(file2, 'rb')
            [s2, s_train2, s_test2, s_valid2] = pickle.load(output)
            output.close()

            for thoI in [2, 3, 4, 5] :

                tho = thoI / 10.0


                s_train = [s_train1[i] * (1 - tho) + s_train2[i] * (tho) for i in xrange(len(s_train1))]
                s_test  = [s_test1[i] * (1 - tho) + s_test2[i] * (tho) for i in xrange(len(s_test1))]
                s_valid = [s_valid1[i] * (1 - tho) + s_valid2[i] * (tho) for i in xrange(len(s_valid1))]


                train_set, valid_set, test_set, dic = load.atisfold(4)
                idx2label = dict((k, v) for v, k in dic['labels2idx'].iteritems())
                idx2word = dict((k, v) for v, k in dic['words2idx'].iteritems())

                train_lex, train_ne, train_y = train_set
                valid_lex, valid_ne, valid_y = valid_set
                test_lex, test_ne, test_y = test_set

                vocsize = len(dic['words2idx'])
                nclasses = len(dic['labels2idx'])
                nsentences = len(train_lex)



                predictions_train = [ [idx2label[x] for x in numpy.argmax(s_p[:, 0, :], axis=1)] for s_p in s_train]
                groundtruth_train = [map(lambda x: idx2label[x], y) for y in train_y]
                words_train = [map(lambda x: idx2word[x], w) for w in train_lex]

                predictions_test = [ [idx2label[x] for x in numpy.argmax(s_p[:, 0, :], axis=1)] for s_p in s_test]
                groundtruth_test = [map(lambda x: idx2label[x], y) for y in test_y]
                words_test = [map(lambda x: idx2word[x], w) for w in test_lex]

                predictions_valid = [ [idx2label[x] for x in numpy.argmax(s_p[:, 0, :], axis=1)] for s_p in s_valid]
                groundtruth_valid = [map(lambda x: idx2label[x], y) for y in valid_y]
                words_valid = [map(lambda x: idx2word[x], w) for w in valid_lex]



                # evaluation // compute the accuracy using conlleval.pl
                res_train = 0
                res_train = conlleval(predictions_train, groundtruth_train, words_train, folder + '/' + id + '.train.txt')
                res_test = conlleval(predictions_test, groundtruth_test, words_test, folder + '/' + id + 'current.test.txt')
                res_valid = conlleval(predictions_valid, groundtruth_valid, words_valid, folder + '/' + id + 'current.valid.txt')

                print tho, '    ',  'train F1', res_train[
                    'f1'], 'valid F1', res_valid[
                    'f1'], 'best test F1', res_test['f1'], ' ' * 20
                if res_valid['f1']  >= best[thoI][0] :
                    best[thoI][0] = res_valid['f1']
                    best[thoI][1] = res_test['f1']
            for index in xrange(len(best)) :
                print 'best ', index , ' valid=' , best[index][0], ' test=' , best[index][1]