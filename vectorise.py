import fasttext
from pymystem3 import Mystem

with open('input.txt') as f:
    raw_input = ''.join(f.readlines())

my = Mystem()

print('Lemmatising...')
lemmas = ''.join(my.lemmatize(raw_input))
with open('input_lemmas.txt', 'w') as f:
    f.write(lemmas)

print('fasttext training started...')
model = fasttext.train_unsupervised('input_lemmas.txt', model='skipgram', dim=300, wordNgrams=5, minCount=1,
                                    bucket=10000000)
print(model.get_labels())
