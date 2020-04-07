from nn.fasttextnn import FastTextNN
import fasttext
import sys

model = fasttext.load_model('result.bin')
print('Nearest neighbours for %s are ...' % sys.argv[1])
fasttext_nn = FastTextNN(model)
print(fasttext_nn.nearest_words(sys.argv[1]))
