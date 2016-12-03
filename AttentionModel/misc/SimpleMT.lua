require 'nn'
local LSTM_decoder = require 'misc.LSTM_decoder'
local LSTM_encoder = require 'misc.LSTM_encoder'
local utils = require 'misc.utils'

local SimpleMT, Parent = torch.class('nn.SimpleMT', 'nn.Module')

function SimpleMT:__init(opt)
	
	Parent.__init(self)
	self.lang1VocabSize = utils.getopt(opt, 'lang1VocabSize')
	self.lang2VocabSize = utils.getopt(opt, 'lang2VocabSize')
	self.wordEmbeddingSize = utils.getopt(opt, 'wordEmbeddingSize', 512)
	self.rnnLayerSize = utils.getopt(opt, 'rnnLayerSize', 512)
	self.numLayers = utils.getopt(opt, 'numLayers', 2)
	self.dropout = utils.getopt(opt, 'dropout', 0.5)
	self.maxLength1 = utils.getopt(opt, 'maxLength1') -- maximum length of lang1 sentences
	self.maxLength2 = utils.getopt(opt, 'maxLength2') -- maximum length of lang2 sentences
	-- Word embedding layers for the 2 languages
	self.lang1Embedding = nn.LookupTable(self.lang1VocabSize+1, self.wordEmbeddingSize)
	self.lang2Embedding = nn.LookupTable(self.lang2VocabSize+1, self.wordEmbeddingSize)
	-- Lazy initialization of init states, helpful in conversion to CUDA tensor
	self:initializeStates(1)
	-- To encode the 1st langugage sentence
	self.Encoder = LSTM_encoder.lstm(self.wordEmbeddingSize, self.rnnLayerSize, self.numLayers, self.dropout)
	-- To decode to the 2nd language sentence
	self.Decoder = LSTM_decoder.lstm(self.wordEmbeddingSize, self.lang2VocabSize+1, self.rnnLayerSize, self.numLayers, self.dropout)

end

function SimpleMT:initializeStates(batchSize)
	assert(batchSize ~= nil, 'No batchSize provided!')
	if not self.EncoderInitStates then self.EncoderInitStates = {} end
	if not self.DecoderInitStates then self.DecoderInitStates = {} end

	for i = 1, 2*self.numLayers do
		if self.EncoderInitStates[i] then
			if self.EncoderInitStates[i]:size(1) ~= batchSize then
				self.EncoderInitStates[i]:resize(batchSize, self.rnnLayerSize):zero()
			end
		else
			self.EncoderInitStates[i] = torch.zeros(batchSize, self.rnnLayerSize)
		end
		if self.DecoderInitStates[i] then
			if self.DecoderInitStates[i]:size(1) ~= batchSize then
				self.DecoderInitStates[i]:resize(batchSize, self.rnnLayerSize):zero()
			end
		else
			self.DecoderInitStates[i] = torch.zeros(batchSize, self.rnnLayerSize)
		end
	end
	self.numStates = 2*self.numLayers
end

function SimpleMT:createClones()

	self.EncoderClones = {self.Encoder}
	self.DecoderClones = {self.Decoder}
	self.lang1EmbeddingClones = {self.lang1Embedding}
	self.lang2EmbeddingClones = {self.lang2Embedding}

	for t = 2, self.maxLength1 do
		print('Encoder clone: t = ' .. t)
		self.EncoderClones[t] = self.Encoder:clone('weight', 'bias', 'gradWeight', 'gradBias')
		self.lang1EmbeddingClones[t] = self.lang1EmbeddingClones[1]:clone('weight', 'gradWeight')
	end

	for t = 2, self.maxLength2+1 do
		print('Decoder clone: t = ' .. t)
		self.DecoderClones[t] = self.Decoder:clone('weight', 'bias', 'gradWeight', 'gradBias')
		self.lang2EmbeddingClones[t] = self.lang2EmbeddingClones[1]:clone('weight', 'gradWeight')
	end

end

function SimpleMT:parameters()

	local p1,g1 = self.Encoder:parameters()
	local p2,g2 = self.Decoder:parameters()
	local p3,g3 = self.lang1Embedding:parameters()
	local p4,g4 = self.lang2Embedding:parameters()

	local params = {}
	for k,v in pairs(p1) do table.insert(params, v) end
	for k,v in pairs(p2) do table.insert(params, v) end
	for k,v in pairs(p3) do table.insert(params, v) end
	for k,v in pairs(p4) do table.insert(params, v) end

	local gradParams = {}
	for k,v in pairs(g1) do table.insert(gradParams, v) end
	for k,v in pairs(g2) do table.insert(gradParams, v) end
	for k,v in pairs(g3) do table.insert(gradParams, v) end
	for k,v in pairs(g4) do table.insert(gradParams, v) end

	return params, gradParams

end

function SimpleMT:training()
	
	if self.EncoderClones == nil or self.DecoderClones == nil then self:createClones() end
	for k,v in pairs(self.EncoderClones) do v:training() end
	for k,v in pairs(self.DecoderClones) do v:training() end
	for k,v in pairs(self.lang1EmbeddingClones) do v:training() end
	for k,v in pairs(self.lang2EmbeddingClones) do v:training() end

end

function SimpleMT:getModulesList()
	return {self.Encoder, self.Decoder, self.lang1Embedding, self.lang2Embedding}
end

function SimpleMT:evaluate()
	
	if self.EncoderClones == nil or self.DecoderClones == nil then self:createClones() end
	for k,v in pairs(self.EncoderClones) do v:evaluate() end
	for k,v in pairs(self.DecoderClones) do v:evaluate() end
	for k,v in pairs(self.lang1EmbeddingClones) do v:evaluate() end
	for k,v in pairs(self.lang2EmbeddingClones) do v:evaluate() end

end

-- Input is a table consisting of 1 element
-- [1] lang1Vector : maxLength1 x batchSize
function SimpleMT:sample(input)

	lang1Vector = input
	local batchSize = lang1Vector:size(2)
	self:initializeStates(batchSize)

	local lang2Pred = torch.Tensor(self.maxLength2+1, batchSize):zero()
	local lang2LogProbs = torch.Tensor(self.maxLength2+1, batchSize):zero()

	-- Encoder forward propagation
	local EncoderState = self.EncoderInitStates
	for t = 1, self.maxLength1 do
		local it = lang1Vector[t]
		
		local skip = 0
		if torch.sum(it) == 0 then
			skip = 1
		end

		if skip == 0 then
			it[it:eq(0)] = self.lang1VocabSize+1
			xt = self.lang1EmbeddingClones[t]:forward(it)
			EncoderState = self.EncoderClones[t]:forward({xt, unpack(EncoderState)})
		end
	end

	-- Initialize DecoderState as the final EncoderState
	local DecoderState = EncoderState
	-- Decoder forward propagation
	for t = 1, self.maxLength2+1 do
		
		local it
		if t == 1 then
			it = torch.Tensor(batchSize):fill(self.lang2VocabSize+1)
		else
			it = lang2Pred[t-1]
		end
	
		xt = self.lang2EmbeddingClones[t]:forward(it)
		local DecoderOut = self.DecoderClones[t]:forward({xt, unpack(DecoderState)})
		DecoderState = {}
		for i = 1, self.numStates do
			table.insert(DecoderState, DecoderOut[i])
		end
		local logsoft = DecoderOut[self.numStates+1]
		-- argmax sampling
		local logprob, pred = torch.max(logsoft, 2)
		lang2LogProbs[t] = logprob:float()
		lang2Pred[t] = pred:float()
	end
		
	return lang2Pred, lang2LogProbs

end

-- Input is a table consisting of 2 elements
-- [1] lang1Vector : maxLength1 x batchSize
-- [2] leng2Vector : maxLength2 x batchSize
function SimpleMT:updateOutput(input)

	lang1Vector = input[1]
	lang2Vector = input[2]
	self.EncoderInputs = {}
	self.DecoderInputs = {}
	self.EncoderStates = {}
	self.DecoderStates = {}
	self.lang1EmbeddingInputs = {}
	self.lang2EmbeddingInputs = {}
	self.tmax1 = 0 
	self.tmax2 = 0

	local batchSize = lang1Vector:size(2)
	-- Initialize the states if not initialized already
	self:initializeStates(batchSize)
	self.EncoderStates[0] = self.EncoderInitStates

	-- Encoder forward propagation
	for t = 1, self.maxLength1 do
		local it = lang1Vector[t]
		local skip = 0
		if torch.sum(it) == 0 then
			skip = 1
		end

		if skip == 0 then -- for optimization
			it[it:eq(0)] = self.lang1VocabSize+1  -- To ensure that the LookupTable
			-- doesn't throw errors. 
			self.lang1EmbeddingInputs[t] = it
			xt = self.lang1EmbeddingClones[t]:forward(it)
			self.EncoderInputs[t] = {xt, unpack(self.EncoderStates[t-1])}
			self.EncoderStates[t] = self.EncoderClones[t]:forward(self.EncoderInputs[t])
			self.tmax1 = t
		end
	end
	
	if not self.outputDec then
		self.outputDec = torch.Tensor(self.maxLength2+1, batchSize, self.lang2VocabSize+1)
		if lang2Vector:type() == 'torch.CudaTensor' then
			self.outputDec = self.outputDec:cuda()
		end
	else
		self.outputDec:resize(self.maxLength2+1, batchSize, self.lang2VocabSize+1)
	end

	-- Decoder forward propagation
	-- Initialize decoder states with the encoder final states
	self.DecoderStates[0] = self.EncoderStates[self.tmax1]
	for t = 1, self.maxLength2+1 do -- +1 because first token is the start token
		local it
		if t == 1 then
			-- start and end tokens are the vocab+1 index
			it = torch.Tensor(batchSize):fill(self.lang2VocabSize+1)
		else
			it = lang2Vector[t-1]
		end
		
		local skip = 0
		if torch.sum(it) == 0 then
			skip = 1
		end

		if skip == 0 then
			it[it:eq(0)] = self.lang2VocabSize+1

			self.lang2EmbeddingInputs[t] = it
			xt = self.lang2EmbeddingClones[t]:forward(it)
			self.DecoderInputs[t] = {xt, unpack(self.DecoderStates[t-1])}
			local tempOut = self.DecoderClones[t]:forward(self.DecoderInputs[t])
			self.outputDec[t] = tempOut[self.numStates+1]
			self.DecoderStates[t] = {}
			for i = 1, self.numStates do
				self.DecoderStates[t][i] = tempOut[i]
			end
			self.tmax2 = t
		end
	end

	return self.outputDec

end

-- input: 2 tensors
-- [1]: maxLength1 x batchSize
-- [2]: maxLength2 x batchSize
-- gradOutput: maxLength2 x batchSize
function SimpleMT:updateGradInput(input, gradOutput)
	
	self.dDecoderStates = {[self.tmax2]=self.DecoderInitStates} -- just initialize as zeros 
	self.dEncoderStates = {}
	-- Decoder backward propagation
	for t = self.tmax2, 1, -1 do
		
		local dDecoder = {}
		for i = 1, self.numStates do
			table.insert(dDecoder, self.dDecoderStates[t][i])
		end
		table.insert(dDecoder, gradOutput[t])
		
		local dInputs = self.DecoderClones[t]:backward(self.DecoderInputs[t], dDecoder)
		self.dDecoderStates[t-1] = {}
		-- gradients wrt prev_c, prev_h
		for i = 2, self.numStates+1 do
			table.insert(self.dDecoderStates[t-1], dInputs[i])
		end
		-- gradients wrt wordEmbedding
		local it = self.lang2EmbeddingInputs[t]
		dw = self.lang2EmbeddingClones[t]:backward(it, dInputs[1])
	end

	-- Initialize encoder gradients from the first decoder gradient
	self.dEncoderStates[self.tmax1] = self.dDecoderStates[0]
	-- Encoder backward propagation
	for t = self.tmax1, 1, -1 do
		local dEncoder = {}
		for i = 1, self.numStates do
			table.insert(dEncoder, self.dEncoderStates[t][i])
		end

		local dInputs = self.EncoderClones[t]:backward(self.EncoderInputs[t], dEncoder)
		
		self.dEncoderStates[t-1] = {}
		-- gradients wrt prev_C, prev_h
		for i = 2, self.numStates+1 do
			table.insert(self.dEncoderStates[t-1], dInputs[i])
		end
		-- gradients wrt wordEmbedding
		local it = self.lang1EmbeddingInputs[t]
		dw = self.lang1EmbeddingClones[t]:backward(it, dInputs[1])
	end

	self.gradInput = {torch.Tensor()}
	return self.gradInput

end

-------------------------------------------------------------------------------
-- Language Model-aware Criterion
-------------------------------------------------------------------------------

local crit, parent = torch.class('nn.LanguageModelCriterion', 'nn.Criterion')
function crit:__init()
  parent.__init(self)
end

--[[
input is a Tensor of size (D+1)xNx(M+1)
seq is a LongTensor of size DxN. The way we infer the target
in this criterion is as follows:
- at last time step the output is always the special END token (last dimension)
The criterion must be able to accomodate variably-sized sequences by making sure
the gradients are properly set to zeros where appropriate.
--]]
function crit:updateOutput(input, seq)
  self.gradInput:resizeAs(input):zero() -- reset to zeros
  local L,N,Mp1 = input:size(1), input:size(2), input:size(3)
  local D = seq:size(1)
  assert(D == L-1, 'input Tensor should be 1 larger in time')

  local loss = 0
  local n = 0
  for b=1,N do -- iterate over batches
    local first_time = true
    for t=1,L do -- iterate over sequence time, t = 1 is where the start sequence begins 
      -- fetch the index of the next token in the sequence
      local target_index
      if t > D then -- we are out of bounds of the index sequence: pad with null tokens
        target_index = 0
      else
        target_index = seq[{t,b}] -- t is correct, since at t=1 START token was fed in and we want to predict first word (and 2-1 = 1).
      end
      -- the first time we see null token as next index, actually want the model to predict the END token
      if target_index == 0 and first_time then
        target_index = Mp1
        first_time = false
      end

      -- if there is a non-null next token, enforce loss!
      if target_index ~= 0 then
        -- accumulate loss
        loss = loss - input[{ t,b,target_index }] -- log(p)
        self.gradInput[{ t,b,target_index }] = -1
        n = n + 1
      end

    end
  end
  self.output = loss / n -- normalize by number of predictions that were made
  self.gradInput:div(n)
  return self.output
end

function crit:updateGradInput(input, seq)
  return self.gradInput
end

