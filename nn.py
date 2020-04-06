from nn.fasttextnn import FastTextNN
import fasttext

model = fasttext.load_model('result.bin')
print('Model loaded...')
fasttext_nn = FastTextNN(model)
print(fasttext_nn.nearest_words('азарт'))
