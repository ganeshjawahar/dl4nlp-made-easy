--[[

Skip-gram with negative sampling (Skip-Gram)

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

-- Step 1: Define your vocabulary map
vocab={}
vocab['i']=1
vocab['like']=2
vocab['nlp']=3
vocab['hate']=4
vocab['dl']=5

-- Step 2: Define constants
vocab_size=5
word_embed_size=10
learning_rate=0.01
window_size=2
max_epochs=5
neg_samples_per_pos_word=1

-- Step 3: Prepare your dataset
word1=torch.Tensor{vocab['like']}
context1=torch.Tensor{vocab['i'],vocab['nlp'],vocab['dl']} -- P(i, nlp | like) (Note: 'dl' is a sample negative context for 'like')
label1=torch.Tensor({1,1,0}) -- 0 denotes negative samples; 1 denotes the positve pairs
word2=torch.Tensor{vocab['hate']}
context2=torch.Tensor{vocab['i'],vocab['dl'],vocab['nlp']} -- P(i, dl | hate) (Note: 'nlp' is a sample negative context for 'hate')
label2=torch.Tensor({1,1,0}) 
dataset={}
function dataset:size() return 2 end
dataset[1]={{context1,word1},label1}
dataset[2]={{context2,word2},label2}

-- Step 4: Define your model
wordLookup=nn.LookupTable(vocab_size,word_embed_size)
contextLookup=nn.LookupTable(vocab_size,word_embed_size)
model=nn.Sequential()
model:add(nn.ParallelTable())
model.modules[1]:add(contextLookup)
model.modules[1]:add(wordLookup)
model:add(nn.MM(false,true)) -- 'true' to transpose the word embeddings before matrix multiplication
model:add(nn.Sigmoid())

-- Step 5: Define the loss function (Binary cross entropy error)
criterion=nn.BCECriterion()

-- Step 6: Define the trainer
trainer=nn.StochasticGradient(model,criterion)
trainer.learningRate=learning_rate
trainer.maxIteration=max_epochs

print('Word Lookup before learning')
print(wordLookup.weight)

-- Step 7: Train the model with dataset
trainer:train(dataset)

-- Step 8: Get the word embeddings
print('\nWord Lookup after learning')
print(wordLookup.weight)