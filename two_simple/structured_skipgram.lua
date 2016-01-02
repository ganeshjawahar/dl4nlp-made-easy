--[[

--------------------
Structured Skip-Gram
--------------------

References:
1. http://www.cs.cmu.edu/~lingwang/papers/naacl2015.pdf

]]--

require 'torch'
require 'nn'
require 'optim'
require 'rnn'

--[[
Dataset:
i like nlp
i hate dl !
]]--

-- Define your vocabulary map
vocab = {}
vocab['i'] = 1
vocab['like'] = 2
vocab['nlp'] = 3
vocab['hate'] = 4
vocab['dl'] = 5
vocab['!'] = 6

-- Define constants
vocab_size = 6 -- number of words in the vocabulary
word_embed_size = 10 -- size of word embedding you are looking for
learning_rate = 0.01 -- initial learning rate for the training
window_size = 2 -- no. of surrounding words to predict. 2 means left and right word
max_epochs = 5 -- number of complete passes of the training set
l2_reg = 0.001 -- regularization parameter for L2-norm

-- Prepare your dataset
word_1 = torch.Tensor{vocab['like']}
context_1 = torch.Tensor{vocab['i'], vocab['nlp']} -- P(i, nlp | like)
word_2 = torch.Tensor{vocab['hate']}
context_2 = torch.Tensor{vocab['i'], vocab['dl']} -- P(i, dl | hate)
word_3 = torch.Tensor{vocab['dl']}
context_3 = torch.Tensor{vocab['hate'], vocab['!']} -- P(hate, ! | dl)
dataset = {}
dataset[1] = {word_1, context_1}
dataset[2] = {word_2, context_2}
dataset[3] = {word_3, context_3}

-- Define your model
word_lookup = nn.LookupTable(vocab_size, word_embed_size)
context_model = nn.Sequential()
context_model:add(nn.Linear(word_embed_size, vocab_size))
context_model:add(nn.LogSoftMax())
context_model_2 = context_model:clone() -- cloning the context model lookup and NOT sharing the layer parameters.
model = nn.Sequential()
model:add(word_lookup) -- first layer consumes the word index and outputs the emebedding
model:add(nn.ConcatTable()) -- branches the input to 'window_size' paths
model.modules[2]:add(context_model) -- first context path
model.modules[2]:add(context_model_2) -- second context path

-- Define the loss function (Negative log-likelihood)
criterion = nn.SequencerCriterion(nn.ClassNLLCriterion()) -- To accomodate multiple output distributions, we go for sequencer.

-- Define the trainer
trainer = nn.StochasticGradient(model,criterion)
trainer.learningRate = learning_rate
trainer.maxIteration = max_epochs

print('Word Lookup before learning')
print(word_lookup.weight)

-- Train the model with dataset
params, grad_params = model:getParameters()
feval = function(x)
	-- Get new params
	params:copy(x)

	-- Reset gradients (buffers)
	grad_params:zero()

	-- loss is average of all criterions
	local loss = 0
	for i = 1, #dataset do
		local output = model:forward(dataset[i][1])
		loss = loss + criterion:forward(output, dataset[i][2])
		local grads = criterion:backward(output, dataset[i][2])
		model:backward(dataset[i][1], grads)
	end
	grad_params:div(#dataset)

	-- L2 regularization
	loss = loss + 0.5 * l2_reg * (params:norm() ^ 2)

	return loss, grad_params
end
optim_state = {learningRate = learning_rate}
print('# StochasticGradient: training')
local l = 0
for epoch = 1, max_epochs do
	local _, loss = optim.sgd(feval, params, optim_state)
	l = loss[1]
	print('# current error = '..l)
end
print('# StochasticGradient: you have reached the maximum number of iterations')
print('# training error = '..l)

-- Get the word embeddings
print('\nWord Lookup after learning')
print(word_lookup.weight)