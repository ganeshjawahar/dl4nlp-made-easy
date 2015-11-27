--[[

--------------------------------------------
Skip-gram with negative sampling (Skip-Gram)
--------------------------------------------

References:
1. http://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf
2. http://cs224d.stanford.edu/lecture_notes/LectureNotes1.pdf

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
neg_samples_per_pos_word = 1 -- no of negative context for every positive (word, context) pair

-- Prepare your dataset
word_1 = torch.Tensor{vocab['like']}
context_1 = torch.Tensor{vocab['i'], vocab['nlp'], vocab['dl']} -- P(i, nlp | like) (Note: 'dl' is a sample negative context for 'like')
label_1 = torch.Tensor({1, 1, 0}) -- 0 denotes negative samples; 1 denotes the positve pairs
word_2 = torch.Tensor{vocab['hate']}
context_2 = torch.Tensor{vocab['i'], vocab['dl'], vocab['nlp']} -- P(i, dl | hate) (Note: 'nlp' is a sample negative context for 'hate')
label_2 = torch.Tensor({1, 1, 0}) 
dataset = {}
dataset[1] = {{context_1, word_1}, label_1}
dataset[2] = {{context_2, word_2}, label_2}
function dataset:size() return 2 end -- define the number of input samples in your dataset (which is 2)

-- Define your model
word_lookup = nn.LookupTable(vocab_size, word_embed_size)
context_lookup = nn.LookupTable(vocab_size, word_embed_size)
model = nn.Sequential()
model:add(nn.ParallelTable()) -- when the input is a table of tensors (context and word), we need parallel table.
model.modules[1]:add(context_lookup) -- consumes context word indices, stacks the context embeddings in a matrix and outputs it.
model.modules[1]:add(word_lookup) -- consumes target word index, and outputs the embedding.
model:add(nn.MM(false, true)) -- 'true' to transpose the word embedding before matrix multiplication
model:add(nn.Sigmoid()) -- this non-linearity keeps the output between 0 and 1 (ideal for our 2-class problem)

-- Define the loss function (Binary cross entropy error)
criterion = nn.BCECriterion()

-- Define the trainer
trainer = nn.StochasticGradient(model,criterion)
trainer.learningRate = learning_rate
trainer.maxIteration = max_epochs

print('Word Lookup before learning')
print(word_lookup.weight)

-- Train the model with dataset
trainer:train(dataset)

-- Get the word embeddings
print('\nWord Lookup after learning')
print(word_lookup.weight)