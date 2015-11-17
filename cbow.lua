--[[

Continous Bag-of-words model (CBOW)

1. http://arxiv.org/pdf/1301.3781.pdf
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
vocab={}
vocab['i']=1
vocab['like']=2
vocab['nlp']=3
vocab['hate']=4
vocab['dl']=5

-- Define constants
vocab_size=5
word_embed_size=10
learning_rate=0.01
window_size=2
max_epochs=5

-- Prepare your dataset
input1=torch.Tensor{vocab['i'],vocab['nlp']} -- P(like | i, nlp)
output1=torch.Tensor{vocab['like']}
input2=torch.Tensor{vocab['i'],vocab['dl']} -- P(hate | i, dl)
output2=torch.Tensor{vocab['hate']}
dataset={}
function dataset:size() return 2 end
dataset[1]={input1,output1}
dataset[2]={input2,output2}

-- Define your model
model=nn.Sequential()
model:add(nn.LookupTable(vocab_size,word_embed_size))
model:add(nn.Mean())
model:add(nn.Linear(word_embed_size,vocab_size))
model:add(nn.LogSoftMax())

-- Define the loss function (Negative log-likelihood)
criterion=nn.ClassNLLCriterion()

-- Define the trainer
trainer=nn.StochasticGradient(model,criterion)
trainer.learningRate=learning_rate
trainer.maxIteration=max_epochs

print('Word Lookup before learning')
print(model.modules[1].weight)

-- Train the model with dataset
trainer:train(dataset)

--Get the word embeddings
print('\nWord Lookup after learning')
print(model.modules[1].weight)