import theano
import numpy
import os
import math
from itertools import *

from theano import tensor as T
from collections import OrderedDict
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from theano.ifelse import ifelse


def getWeight(dim):
    return theano.shared(math.sqrt(6.0 / numpy.sum(dim)) * numpy.random.uniform(-1.0, 1.0, dim).astype(theano.config.floatX))
def getBias(dim) :
    return theano.shared(numpy.zeros(dim, dtype=theano.config.floatX))


def updates_func(parameters, gradients):
    rou = 0.95
    eps = 1e-6
    gradients_sq = [theano.shared(numpy.zeros(p.get_value().shape, dtype=theano.config.floatX)) for p in parameters]
    deltas_sq = [theano.shared(numpy.zeros(p.get_value().shape, dtype=theano.config.floatX)) for p in parameters]
    # calculates the new "average" delta for the next iteration
    gradients_sq_new = [rou * g_sq + (1 - rou) * (g ** 2) for g_sq, g in izip(gradients_sq, gradients)]

    # calculates the step in direction. The square root is an approximation to getting the RMS for the average value
    deltas = [(T.sqrt(d_sq + eps) / T.sqrt(g_sq + eps)) * grad for d_sq, g_sq, grad in izip(deltas_sq, gradients_sq_new, gradients)]

    # calculates the new "average" deltas for the next step.
    deltas_sq_new = [rou * d_sq + (1 - rou) * (d ** 2) for d_sq, d in izip(deltas_sq, deltas)]

    # Prepare it as a list f
    gradient_sq_updates = zip(gradients_sq, gradients_sq_new)
    deltas_sq_updates = zip(deltas_sq, deltas_sq_new)
    parameters_updates = [(p, p - d) for p, d in izip(parameters, deltas)]
    return gradient_sq_updates + deltas_sq_updates + parameters_updates


class model(object):
    def __init__(self, nh, nc, ne, de, attention, h_win, lvrg, wv=None):
        '''
        nh :: dimension of the hidden layer
        nc :: number of classes
        ne :: number of word embeddings in the vocabulary
        de :: dimension of the word embeddings
        cs :: word window context size
        '''


        # parameters of the model
        EDMode = 'ED'
        iParams = {}

        if wv is None:
            emb = getWeight((ne + 1, de))
        else:
            emb = theano.shared(wv.astype(theano.config.floatX))
        # iParams['emb'] = getWeight((ne + 1, de))

        # decoder
        for lr in "lr":
            for ed in EDMode:
                if ed != 'D':
                    iParams['h0' + ed + lr] = getBias((nh))
                    iParams['C0' + ed + lr] = getBias((nh))
                iParams['Wi' + ed + lr] = getWeight((de, nh))
                iParams['Wf' + ed + lr] = getWeight((de , nh))
                iParams['Wc' + ed + lr] = getWeight((de , nh))
                iParams['Wo' + ed + lr] = getWeight((de, nh))

                iParams['Ui' + ed + lr] = getWeight((nh, nh))
                iParams['Uf' + ed + lr] = getWeight((nh, nh))
                iParams['Uc' + ed + lr] = getWeight((nh, nh))
                iParams['Uo' + ed + lr] = getWeight((nh, nh))

                iParams['Vo' + ed + lr] = getWeight((nh, nh))

                iParams['bi' + ed + lr] = getBias((nh))
                iParams['bf' + ed + lr] = getBias((nh))
                iParams['bc' + ed + lr] = getBias((nh))
                iParams['bo' + ed + lr] = getBias((nh))

            # prediction
            if attention != 'none':

                #iParams['lp' + lr] = getWeight((h_win[0], nh ))
                #iParams['rp' + lr] = getWeight((h_win[1], nh))
                lp = getWeight((h_win[0], nh ))
                rp = getWeight((h_win[1], nh ))

                if attention == 'dot':
                    pass
                if attention == 'general':
                    iParams['Wa' + lr] = getWeight((nh, nh))
                if attention == 'concat':
                    iParams['WaE' + lr] = getWeight((nh, nh))
                    iParams['WaD' + lr] = getWeight((nh , nh))
                    iParams['Wab' + lr] = getBias((nh))
                    iParams['Va' + lr] = getWeight((nh))


            iParams['Wcs' + lr] = getWeight((nh ,nc))
            iParams['Whs' + lr] = getWeight((nh ,nc))
            #iParams['Wcb' + lr] = getBias((nc))
            #iParams['Whb' + lr] = getBias((nc))
            iParams['Wchb' + lr] = getBias((nc))
        self.srng = RandomStreams(seed=12345)

        self.params = []
        for k, v in iParams.items():
            self.params.append(v)

        # bundle
        dropRate = T.scalar('dropRate', theano.config.floatX)
        useDrop = T.scalar('useDrop', theano.config.floatX)

        def nor(m):
            return m * T.cast(m.shape[0], 'float32') / (1.0 * T.sum(m))

        def drop(layer):
            mask = self.srng.binomial(layer.shape,
                                      p=(1 - dropRate), n=1,
                                      dtype=layer.dtype)

            m, _ = theano.scan(fn=nor, \
                                  sequences=mask, \
                                  n_steps=mask.shape[0])
            return useDrop * (layer * m) + (1 - useDrop) * layer

        # real thing start
        idxs = T.imatrix()  # as many columns as context window size/lines as words in the sentence
        # sub_emb = iParams['emb'][idxs.flatten(1)]
        sub_emb = emb[idxs.flatten(1)]
        x = sub_emb.reshape((idxs.shape[0], de))
        y = T.ivector()  # label

        stt = {}
        for lr in "lr":
            h = {}
            for ed in EDMode:
                # o = iParams['h0' + ed + lr]
                if ed != 'D':
                    hInit = iParams['h0'+ ed + lr]
                    CInit = iParams['C0'+ ed + lr]
                else:
                    if lr == 'l':
                        hInit = h['E'][-1]
                        CInit = C[-1]
                    if lr == 'r':
                        hInit = h['E'][0]
                        CInit = C[0]

                def hh(x_t, h_tm1, C_tm1):
                    i = T.nnet.sigmoid(T.dot(x_t, iParams['Wi' + ed + lr]) + T.dot(h_tm1, iParams['Ui' + ed + lr]) + iParams['bi' + ed + lr])
                    Cba = T.tanh(T.dot(x_t, iParams['Wc' + ed + lr]) + T.dot(h_tm1, iParams['Uc' + ed + lr]) + iParams['bc' + ed + lr])
                    f = T.nnet.sigmoid(T.dot(x_t, iParams['Wf' + ed + lr]) + T.dot(h_tm1, iParams['Uf' + ed + lr]) + iParams['bf' + ed + lr])

                    C = i * Cba + f * C_tm1

                    o = T.nnet.sigmoid(T.dot(x_t, iParams['Wo' + ed + lr]) + T.dot(h_tm1, iParams['Uo' + ed + lr])
                                       + T.dot(C, iParams['Vo' + ed + lr]) + iParams['bo' + ed + lr])

                    h_t = o * T.tanh(C)


                    return [h_t, C] # [h_t * 0 + i, C]

                [h[ed], C], _ = theano.scan(fn=hh, \
                                                sequences=drop(x), outputs_info=[hInit, CInit], \
                                                n_steps=x.shape[0],
                                                go_backwards=(lr == 'r'))
                if lr == 'r':
                    h[ed] = h[ed][::-1]

            pHl = h['E']
            #pHl = T.concatenate((iParams['lp' + lr], pHl), axis=0)
            #pHl = T.concatenate((pHl, iParams['rp'+ lr] ), axis=0)
            pHl = T.concatenate((lp, pHl ), axis=0)
            pHl = T.concatenate((pHl, rp), axis=0)

            def concat_r(hec, hdc):
                t = T.tanh(T.dot(hec, iParams['WaE' + lr]) + T.dot(hdc, iParams['WaD' + lr]) + iParams['Wab' + lr])
                return T.dot(t, T.transpose(iParams['Va' + lr]))
            def r_c(ii, hd, heL):

                hehL = T.concatenate((heL[ii: ii + h_win[0]], heL[ii + h_win[0] + lvrg: ii + h_win[0] + 1 + h_win[1]]), axis=0)
                # attention
                if attention == 'none':
                    c = hd
                else:
                    if attention == 'dot' :
                        score = T.dot(hd, T.transpose((hehL)))
                    if attention == 'general':
                        v = T.dot(hd, iParams['Wa' + lr])
                        score = T.dot(v, T.transpose((hehL)))
                    if attention == 'concat':
                        score, _ = theano.scan(fn=concat_r, \
                               sequences=(hehL), \
                               non_sequences=hd,
                               n_steps=hehL.shape[0])
                    # score = score * 0 + 10
                    a = T.nnet.softmax(score)[:][0]
                    c = T.dot(a, hehL)
                return c, a

            [cL, aaL], _ = theano.scan(fn=r_c, \
                                           sequences=[T.arange(h['D'].shape[0]), drop(h['D'])], \
                                           non_sequences=(drop(pHl)),
                                           n_steps=h['D'].shape[0])


            def ss(cc, hh):
                s = T.nnet.softmax(T.dot(cc, iParams['Wcs' + lr]) + T.dot(hh, iParams['Whs' + lr]) * lvrg  + iParams['Wchb' + lr])
                #s = s / 2
                return s

            stt[lr], _ = theano.scan(fn=ss, \
                           sequences=[drop(cL), drop(h['E'])], \
                           n_steps=h['E'].shape[0])

        #predict
        sl0 = stt['l'][:, 0, :]
        sr0 = stt['r'][:, 0, :]
        rho = T.ivector()
        def r_rho(r):

            s = sl0 * r / 100.0 + sr0 * (1 - r / 100.0)
            p_y_given_x_sentence = s
            y_p = T.argmax(p_y_given_x_sentence, axis=1)
            return y_p
        y_pred, _ = theano.scan(fn=r_rho, \
                           sequences=rho, \
                           n_steps=rho.shape[0])

        nll = T.sum(T.diag(-T.log(sl0)[:, y])) + T.sum(T.diag(-T.log(sr0)[:, y]))


        #rr all
        #for p in self.params :
        #    nll += 0.001 * T.sum(p ** 2)

        #rr attention
        # nll += 0.001 * T.sum(iParams['Wa'] ** 2)

        gradients = T.grad(nll, self.params)

        # updates = updates_func(self.params, gradients)
        #lr = 0.0627142536696559
        lr = 0.025
        updates = OrderedDict(( p, p-lr*g ) for p, g in zip( self.params , gradients))

        # theano functions
        self.classify = theano.function(inputs=[idxs, dropRate, useDrop, rho], outputs=y_pred)

        self.train = theano.function(inputs=[idxs, y, dropRate, useDrop],
                                     outputs=[nll, aaL],
                                     updates=updates)
