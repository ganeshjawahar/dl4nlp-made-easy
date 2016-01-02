--[[

--------------------------------------------------------------
Distributed Bag of Words version of Paragraph Vector (PV-DBOW)
--------------------------------------------------------------

References:
1. https://cs.stanford.edu/~quocle/paragraph_vector.pdf
2. http://arxiv.org/abs/1507.07998
3. https://cs224d.stanford.edu/reports/HongSeokho.pdf

]]--

require 'torch'
require 'nn'

--[[
Dataset (Train):
how are you
i am awesome
Dataset (Test):
are you okay
]]--

-- Define your vocabulary map
vocab = {}
vocab['how'] = 1
vocab['are'] = 2
vocab['you'] = 3
vocab['i'] = 4
vocab['am'] = 5
vocab['awesome'] = 6
vocab['<unknown>'] = 7 -- substitute this with new words seen during testing

-- Define constants
vocab_size = 7 -- number of words in the vocabulary
word_embed_size = 10 -- size of word embedding you are looking for
doc_embed_size = 10 -- size of document embedding you are looking for
learning_rate = 0.01 -- initial learning rate for the training
max_epochs = 5 -- number of complete passes of the training set
neg_samples_per_pos_word = 1 -- no of negative context for every positive (paragraph_id, context) pair

-- Prepare your dataset
-- First sentence
pid_1 = torch.Tensor{1} -- P(how | <pid>)
context_1 = torch.Tensor{vocab['how'], vocab['am']} -- note: 'am' is a sample negative context for this sentence
output_1 = torch.Tensor{1, 0}
pid_2 = torch.Tensor{1} -- P(are | <pid>)
context_2 = torch.Tensor{vocab['are'], vocab['i']} -- note: 'i' is a sample negative context for this sentence
output_2 = torch.Tensor{1, 0}
pid_3 = torch.Tensor{1} -- P(you | <pid>)
context_3 = torch.Tensor{vocab['you'], vocab['awesome']} -- note: 'awesome' is a sample negative context for this sentence
output_3 = torch.Tensor{1, 0}
-- Second sentence
pid_4 = torch.Tensor{2} -- P(i | <pid>)
context_4 = torch.Tensor{vocab['i'], vocab['you']} -- note: 'you' is a sample negative context for this sentence
output_4 = torch.Tensor{1, 0}
pid_5 = torch.Tensor{2} -- P(am | <pid>)
context_5 = torch.Tensor{vocab['am'], vocab['how']} -- note: 'how' is a sample negative context for this sentence
output_5 = torch.Tensor{1, 0}
pid_6 = torch.Tensor{2} -- P(awesome | <pid>)
context_6 = torch.Tensor{vocab['awesome'], vocab['are']} -- note: 'are' is a sample negative context for this sentence
output_6 = torch.Tensor{1, 0}
train_dataset = {}
train_dataset[1] = {{context_1, pid_1}, output_1}
train_dataset[2] = {{context_2, pid_2}, output_2}
train_dataset[3] = {{context_3, pid_3}, output_3}
train_dataset[4] = {{context_4, pid_4}, output_4}
train_dataset[5] = {{context_5, pid_5}, output_5}
train_dataset[6] = {{context_6, pid_6}, output_6}
function train_dataset:size() return 6 end -- define the number of input samples in your train dataset (which is 6)
-- Test sentence
pid_7 = torch.Tensor{3} -- P(are | <pid>)
context_7 = torch.Tensor{vocab['are'], vocab['i']} -- note: 'i' is a sample negative context for this sentence
output_7 = torch.Tensor{1, 0}
pid_8 = torch.Tensor{3} -- P(you | <pid>)
context_8 = torch.Tensor{vocab['you'], vocab['how']} -- note: 'how' is a sample negative context for this sentence
output_8 = torch.Tensor{1, 0}
pid_9 = torch.Tensor{3} -- P(okay | <pid>) but 'okay' is unknown during training
context_9 = torch.Tensor{vocab['<unknown>'], vocab['awesome']} -- note: 'awesome' is a sample negative context for this sentence
output_9 = torch.Tensor{1, 0}
test_dataset = {}
test_dataset[1] = {{context_7, pid_7}, output_7}
test_dataset[2] = {{context_8, pid_8}, output_8}
test_dataset[3] = {{context_9, pid_9}, output_9}
function test_dataset:size() return 3 end -- define the number of input samples in your test dataset (which is 3)

-- Define your model
word_lookup = nn.LookupTable(vocab_size, word_embed_size)
doc_lookup = nn.LookupTable(2 + 1, doc_embed_size) -- rows is equal to no. of documents in train plus testing.
model = nn.Sequential()
model:add(nn.ParallelTable()) -- when the input is a table of tensors (context and word), we need parallel table.
model.modules[1]:add(word_lookup) -- consumes context word indices, stacks the context embeddings in a matrix and outputs it.
model.modules[1]:add(doc_lookup) -- consumes document index, and outputs the document embedding.
model:add(nn.MM(false, true)) -- 'true' to transpose the word embedding before matrix multiplication
model:add(nn.Sigmoid()) -- this non-linearity keeps the output between 0 and 1 (ideal for our 2-class problem)

-- Define the loss function (Binary cross entropy error)
criterion = nn.BCECriterion()

-- Define the trainer
trainer = nn.StochasticGradient(model, criterion)
trainer.learningRate = learning_rate
trainer.maxIteration = max_epochs

-- Train the model with dataset
trainer:train(train_dataset)

-- Infer the document embeddings for the test sentence. (Poor model - Need to do inference during testing!)
-- Freeze the word vectors (to stop them from getting updated during testing)
word_lookup.accGradParameters = function() end
print('\nDocument representation of test sentence (are you okay)')
print('\nBefore testing, ')
print(doc_lookup.weight[3]) -- test document index
trainer:train(test_dataset)
print('\nAfter testing, ')
print(doc_lookup.weight[3])