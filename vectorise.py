import fasttext
from pymystem3 import Mystem

with open('normie.txt') as f:
    raw_input = ''.join(f.readlines())

my = Mystem()

print('Lemmatising...')
lemmas = ''.join(my.lemmatize(raw_input))
with open('input_lemmas.txt', 'w') as f:
    f.write(lemmas)

print('fasttext training started...')
model = fasttext.train_unsupervised('input_lemmas.txt',
                                    model='skipgram',
                                    dim=300,             # size of word vectors [100]
                                    lr=0.025,            # learning rate [0.05]
                                    ws=6,                # size of the context window [5]
                                    epoch=5,             # number of epochs [5]
                                    loss='softmax',      # loss function {ns, hs, softmax, ova} [ns]
                                    t=1e-4,              # sampling threshold [0.0001]
                                    lrUpdateRate=100,    # change the rate of updates for the learning rate [100]
                                    wordNgrams=7,        # max length of word ngram [1]
                                    minn=3,              # min length of char ngram [3]
                                    maxn=6,              # max length of char ngram [6]
                                    minCount=1,          # minimal number of word occurences [5]
                                    thread=8,            # number of threads [number of CPUs]
                                    neg=5,               # number of negatives sampled [5]
                                    bucket=10000000,     # number of buckets [2000000]
                                    verbose=4
                                    )
print(model.get_labels())
model.save_model('result.bin')
print('Model loaded...')
