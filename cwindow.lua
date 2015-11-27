--[[

----------------------
Continous Window model
----------------------

References:
1. http://www.cs.cmu.edu/~lingwang/papers/naacl2015.pdf

]]--

require 'torch'
require 'nn'

--[[
Dataset:
i like nlp
i hate dl
]]--

-- Define your vocabulary map
vocab = {}
vocab['i'] = 1
vocab['like'] = 2
vocab['nlp'] = 3
vocab['hate'] = 4
vocab['dl'] = 5

-- Define constants
vocab_size = 5 -- number of words in the vocabulary
word_embed_size = 10 -- size of word embedding you are looking for
learning_rate = 0.01 -- initial learning rate for the training
window_size = 2 -- no. of surrounding words to predict. 2 means left and right word
max_epochs = 5 -- number of complete passes of the training set

-- Prepare your dataset
input_1 = torch.Tensor{vocab['i'], vocab['nlp']} -- P(like | i, nlp)
output_1 = torch.Tensor{vocab['like']}
input_2 = torch.Tensor{vocab['i'], vocab['dl']} -- P(hate | i, dl)
output_2 = torch.Tensor{vocab['hate']}
dataset = {}
dataset[1] = {input_1, output_1}
dataset[2] = {input_2, output_2}
function dataset:size() return 2 end -- define the number of input samples in your dataset (which is 2)

-- Define your model
model = nn.Sequential()
model:add(nn.LookupTable(vocab_size, word_embed_size)) -- consumes the word indices and outputs the embeddings
model:add(nn.View(window_size * word_embed_size)) -- concatenate the context to capture the word order.
model:add(nn.Linear(window_size * word_embed_size, vocab_size)) -- projects concatenated context to |V| size representation.
model:add(nn.LogSoftMax()) -- converts the representation to probability distribution

-- Define the loss function (Negative log-likelihood)
criterion = nn.ClassNLLCriterion()

-- Define the trainer
trainer = nn.StochasticGradient(model, criterion)
trainer.learningRate = learning_rate
trainer.maxIteration = max_epochs

print('Word Lookup before learning')
print(model.modules[1].weight)

-- Train the model with dataset
trainer:train(dataset)

-- Get the word embeddings
print('\nWord Lookup after learning')
print(model.modules[1].weight)