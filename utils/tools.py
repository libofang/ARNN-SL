import random

def shuffle(lol, seed):
    '''
    lol :: list of list as input
    seed :: seed the shuffling

    shuffle inplace each list in the same order
    '''
    for l in lol:
        random.seed(seed)
        random.shuffle(l)

def minibatch(l, bs):
    '''
    l :: list of word idxs
    return a list of minibatches of indexes
    which size is equal to bs
    border cases are treated as follow:
    eg: [0,1,2,3] and bs = 3
    will output:
    [[0],[0,1],[0,1,2],[1,2,3]]
    '''
    out = [[l[j] if j >= 0 and j < len(l) else [-1] * len(l[0]) for j in xrange(i - bs[0], i + 1 + bs[1])
            #if j >= 0
            ]for i in xrange(len(l))]
    #out  = [l[:i] for i in xrange(1, min(bs,len(l)+1) )]
    #out += [l[i-bs:i] for i in xrange(bs,len(l)+1) ]
    assert len(l) == len(out)
    return out

def contextwin(l):
    '''
    win :: int corresponding to the size of the window
    given a list of indexes composing a sentence
    it will return a list of list of indexes corresponding
    to context windows surrounding each word in the sentence

    this function is totally messed.  but still working.
    '''
    l = list(l)

    out = [ l[i:i + 1] for i in range(len(l)) ]

    assert len(out) == len(l)
    return out

def original_contextwin(l, win):
    '''
    win :: int corresponding to the size of the window
    given a list of indexes composing a sentence
    it will return a list of list of indexes corresponding
    to context windows surrounding each word in the sentence
    '''
    assert (win % 2) == 1
    assert win >=1
    l = list(l)

    lpadded = win/2 * [0] + l + win/2 * [0]
    out = [ lpadded[i:i+win] for i in range(len(l)) ]

    assert len(out) == len(l)
    return out