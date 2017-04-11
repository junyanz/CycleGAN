--------------------------------------------------------------------------------
-- Subclass of BaseDataLoader that provides data from two datasets.
-- The samples from the datasets are aligned
-- The datasets are of the same size
--------------------------------------------------------------------------------
require 'data.base_data_loader'

local class = require 'class'
data_util = paths.dofile('data_util.lua')

AlignedDataLoader = class('AlignedDataLoader', 'BaseDataLoader')

function AlignedDataLoader:__init(conf)
  BaseDataLoader.__init(self, conf)
  conf = conf or {}
end

function AlignedDataLoader:name()
  return 'AlignedDataLoader'
end

function AlignedDataLoader:Initialize(opt)
  opt.align_data = 1
  self.idx_A = {1, opt.input_nc}
  self.idx_B = {opt.input_nc+1, opt.input_nc+opt.output_nc}
  local nc = 3--opt.input_nc + opt.output_nc
  self.data = data_util.load_dataset('', opt, nc)
end

-- actually fetches the data
-- |return|: a table of two tables, each corresponding to
-- the batch for dataset A and dataset B
function AlignedDataLoader:LoadBatchForAllDatasets()
  local batch_data, path = self.data:getBatch()
  local batchA = batch_data[{ {}, self.idx_A, {}, {} }]
  local batchB = batch_data[{ {}, self.idx_B, {}, {} }]

  return batchA, batchB, path, path
end

-- returns the size of each dataset
function AlignedDataLoader:size(dataset)
  return self.data:size()
end
