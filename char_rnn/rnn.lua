--[[

-----------------------------------------------------------
Recurrent Neural Network for Character-level language model
-----------------------------------------------------------

References:
1. http://arxiv.org/pdf/1506.02078.pdf
2. http://karpathy.github.io/2015/05/21/rnn-effectiveness/
3. http://cs224d.stanford.edu/lecture_notes/LectureNotes4.pdf

]]--

require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
require 'model.OneHot'
require 'model.misc'
local RNN = require 'model.RNN'
local model_utils = require 'model.model_utils'

--[[
Dataset:
i aspire to be research scientist.
--]]

-- Define your vocabulary map
vocab = {}
vocab['i'] = 1
vocab['a'] = 2
vocab['s'] = 3
vocab['p'] = 4
vocab['r'] = 5
vocab['e'] = 6
vocab['t'] = 7
vocab['o'] = 8
vocab['b'] = 9
vocab['c'] = 10
vocab['h'] = 11
vocab['n'] = 12
vocab[' '] = 13
vocab['.'] = 14

-- Define constants
vocab_size = 14 -- number of characters in the vocabulary
seq_len = 11 -- number of timesteps to unroll for
rnn_size = 50 -- size of RNN internal state
num_layers = 3 -- number of layers in the RNN
dropout = 0.5 -- dropout for regularization, used after each RNN hidden layer. 0 = no dropout
batch_size = 3 -- number of sequences to train on in parallel (in this toy code, the no. of training samples is equal to batch size. BEWARE!)
grad_clip = 5 -- clip gradients at this value
learning_rate = 2e-3 -- learning rate

-- Prepare your dataset
input1 = torch.Tensor{vocab['i'], vocab[' '], vocab['a'], vocab['s'], vocab['p'], vocab['i'], vocab['r'], vocab['e'], vocab[' '], vocab['t'], vocab['o']}
output1 = torch.Tensor{vocab[' '], vocab['a'], vocab['s'], vocab['p'], vocab['i'], vocab['r'], vocab['e'], vocab[' '], vocab['t'], vocab['o'], vocab[' ']}
input2 = torch.Tensor{vocab[' '], vocab['b'], vocab['e'], vocab[' '], vocab['r'], vocab['e'], vocab['s'], vocab['e'], vocab['a'], vocab['r'], vocab['c']}
output2 = torch.Tensor{vocab['b'], vocab['e'], vocab[' '], vocab['r'], vocab['e'], vocab['s'], vocab['e'], vocab['a'], vocab['r'], vocab['c'], vocab['h']}
input3 = torch.Tensor{vocab['h'], vocab[' '], vocab['s'], vocab['c'], vocab['i'], vocab['e'], vocab['n'], vocab['t'], vocab['i'], vocab['s'], vocab['t']}
output3 = torch.Tensor{vocab[' '], vocab['s'], vocab['c'], vocab['i'], vocab['e'], vocab['n'], vocab['t'], vocab['i'], vocab['s'], vocab['t'], vocab['.']}
input_sample = torch.zeros(seq_len, batch_size)
input_sample[{{},1}] = input1 -- fill the first column with first input
input_sample[{{},2}] = input2
input_sample[{{},3}] = input3
output_sample = torch.zeros(seq_len, batch_size)
output_sample[{{},1}] = output1
output_sample[{{},2}] = output2
output_sample[{{},3}] = output3

-- Define RNN Model and criterion
protos = {}
protos.rnn = RNN.rnn(vocab_size, rnn_size, num_layers, dropout)
protos.criterion = nn.ClassNLLCriterion()
params, grad_params = model_utils.combine_all_parameters(protos.rnn) -- put the above things into one flattened parameters tensor

-- Define the initial state of the cell/hidden states
init_state = {}
for L = 1, num_layers do
	table.insert(init_state, torch.zeros(batch_size, rnn_size))
end

-- Make a bunch of clones ('seq_length' times)
clones = {}
for name, proto in pairs(protos) do
    clones[name] = model_utils.clone_many_times(proto, seq_len, not proto.parameters)
end

-- Define feval
local init_state_global = clone_list(init_state)
function feval(x)
	if x ~= params then
        params:copy(x)
    end
    grad_params:zero()
    ------------------ get minibatch -------------------
    local x, y = input_sample, output_sample
    ------------------- forward pass -------------------
    local rnn_state = {[0] = init_state_global}
    local predictions = {} -- softmax outputs
    local loss = 0
    for t = 1, seq_len do
        clones.rnn[t]:training() -- make sure we are in correct mode (this is cheap, sets flag)
        local lst = clones.rnn[t]:forward{x[t], unpack(rnn_state[t-1])}
        rnn_state[t] = {}
        for i=1,#init_state do table.insert(rnn_state[t], lst[i]) end -- extract the state, without output
        predictions[t] = lst[#lst] -- last element is the prediction
        loss = loss + clones.criterion[t]:forward(predictions[t], y[t])
    end
    loss = loss / seq_len
    ------------------ backward pass -------------------
    -- initialize gradient at time t to be zeros (there's no influence from future)
    local drnn_state = {[seq_len] = clone_list(init_state, true)} -- true also zeros the clones
    for t = seq_len, 1, -1 do
        -- backprop through loss, and softmax/linear
        local doutput_t = clones.criterion[t]:backward(predictions[t], y[t])
        table.insert(drnn_state[t], doutput_t)
        local dlst = clones.rnn[t]:backward({x[t], unpack(rnn_state[t-1])}, drnn_state[t])
        drnn_state[t-1] = {}
        for k,v in pairs(dlst) do
            if k > 1 then -- k == 1 is gradient on x, which we dont need
                -- note we do k-1 because first item is dembeddings, and then follow the 
                -- derivatives of the state, starting at index 2. I know...
                drnn_state[t-1][k-1] = v
            end
        end
    end
    ------------------------ misc ----------------------
    -- transfer final state to initial state (BPTT) (may not make sense for this toy sample. useful for real world dataset.)
    init_state_global = rnn_state[#rnn_state]
    grad_params:clamp(-grad_clip, grad_clip)
    return loss, grad_params
end

-- Start training using rmsprop
local optim_state = {learningRate = learning_rate} -- has more customizations like decay rate and so on.
local _, loss = optim.rmsprop(feval, params, optim_state)
print('RNN training loss = '..loss[1])