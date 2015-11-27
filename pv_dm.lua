--[[

-----------------------------------------------------
Distributed Memory Model of Paragraph Vectors (PV-DM)
-----------------------------------------------------

References:
1. https://cs.stanford.edu/~quocle/paragraph_vector.pdf
2. http://arxiv.org/abs/1507.07998
3. https://cs224d.stanford.edu/reports/HongSeokho.pdf

]]--

require 'torch'
require 'nn'

--[[
Dataset (Train):
iiit is located at gachibowli
i live in chennai
Dataset (Test):
is iiit in chennai ?
]]--

-- Define your vocabulary map
vocab = {}
vocab['iiit'] = 1
vocab['is'] = 2
vocab['located'] = 3
vocab['at'] = 4
vocab['gachibowli'] = 5
vocab['i'] = 6
vocab['live'] = 7
vocab['in'] = 8
vocab['chennai'] = 9
vocab['<unknown>'] = 10 -- substitute this with new words seen during testing

-- Define constants
vocab_size = 10 -- number of words in the vocabulary
word_embed_size = 10 -- size of word embedding you are looking for
doc_embed_size = 20 -- size of document embedding you are looking for
learning_rate = 0.01 -- initial learning rate for the training
window_size = 2 -- no. of surrounding words to predict. 2 means left and right word
max_epochs = 5 -- number of complete passes of the training set

-- Prepare your dataset
-- First sentence
input_1 = {torch.Tensor{1}, torch.Tensor{vocab['iiit'], vocab['located']}} -- P(is | iiit, located)
output_1 = torch.Tensor{vocab['is']}
input_2 = {torch.Tensor{1}, torch.Tensor{vocab['is'], vocab['at']}} -- P(located | is, at)
output_2 = torch.Tensor{vocab['located']}
input_3 = {torch.Tensor{1}, torch.Tensor{vocab['is'], vocab['at']}} -- P(at | located, gachibowli)
output_3 = torch.Tensor{vocab['at']}
-- Second sentence
input_4 = {torch.Tensor{2}, torch.Tensor{vocab['is'], vocab['at']}} -- P(live | i, in)
output_4 = torch.Tensor{vocab['live']}
input_5 = {torch.Tensor{2}, torch.Tensor{vocab['live'], vocab['chennai']}} -- P(in | live, chennai)
output_5 = torch.Tensor{vocab['in']}
train_dataset = {}
train_dataset[1] = {input_1, output_1}
train_dataset[2] = {input_2, output_2}
train_dataset[3] = {input_3, output_3}
train_dataset[4] = {input_4, output_4}
train_dataset[5] = {input_5, output_5}
function train_dataset:size() return 5 end -- define the number of input samples in your train dataset (which is 5)
-- Test sentence
input_6 = {torch.Tensor{3}, torch.Tensor{vocab['is'], vocab['in']}} -- P(iiit | is, in)
output_6 = torch.Tensor{vocab['iiit']}
input_7 = {torch.Tensor{3}, torch.Tensor{vocab['iiit'], vocab['chennai']}} -- P(in | iiit, chennai)
output_7 = torch.Tensor{vocab['in']}
input_8 = {torch.Tensor{3}, torch.Tensor{vocab['in'], vocab['<unknown>']}} -- P(chennai | in, ?) but '?' is unknown during training
output_8 = torch.Tensor{vocab['chennai']}
test_dataset = {}
test_dataset[1] = {input_6, output_6}
test_dataset[2] = {input_7, output_7}
test_dataset[3] = {input_8, output_8}
function test_dataset:size() return 3 end -- define the number of input samples in your test dataset (which is 3)

-- Define your model
word_lookup = nn.LookupTable(vocab_size, word_embed_size)
word_model = nn.Sequential()
word_model:add(word_lookup)
word_model:add(nn.View(1, window_size * word_embed_size)) -- flatten the context embeddings into one fat vector
doc_lookup = nn.LookupTable(2 + 1, doc_embed_size) -- rows is equal to no. of documents in train plus testing.
model = nn.Sequential()
model:add(nn.ParallelTable()) -- when the input is a table of tensors (context and document), we need parallel table.
model.modules[1]:add(doc_lookup) -- consumes the document index outputs the document embedding
model.modules[1]:add(word_model) -- consumes the word indices outputs the context embeddings
model:add(nn.JoinTable(2)) -- concatenate both the context and document embedding (capturing word order)
model:add(nn.Linear(((2 * word_embed_size) + doc_embed_size), vocab_size)) -- projects to |V| size representation.
model:add(nn.LogSoftMax()) -- converts the representation to probability distribution

-- Define the loss function (Negative log-likelihood)
criterion = nn.ClassNLLCriterion()

-- Define the trainer
trainer = nn.StochasticGradient(model, criterion)
trainer.learningRate = learning_rate
trainer.maxIteration = max_epochs

-- Train the model with dataset
trainer:train(train_dataset)

-- Infer the document embeddings for the test sentence. (Poor model - Need to do inference during testing!)
-- Freeze the word vectors (to stop them from getting updated during testing)
word_lookup.accGradParameters = function() end
print('\nDocument representation of test sentence (is iiit in chennai ?)')
print('\nBefore testing, ')
print(doc_lookup.weight[3]) -- test document index
trainer:train(test_dataset)
print('\nAfter testing, ')
print(doc_lookup.weight[3])