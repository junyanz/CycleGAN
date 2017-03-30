--------------------------------------------------------------------------------
-- Base Class for Providing Data
--------------------------------------------------------------------------------

local class = require 'class'
require 'torch'

BaseDataLoader = class('BaseDataLoader')

function BaseDataLoader:__init(conf)
   conf = conf or {}   
   self.data_tm = torch.Timer()
end

function BaseDataLoader:name()
  return 'BaseDataLoader'
end

function BaseDataLoader:Initialize(opt)  
end

-- actually fetches the data
-- |return|: a table of two tables, each corresponding to 
-- the batch for dataset A and dataset B
function BaseDataLoader:LoadBatchForAllDatasets()
  return {},{},{},{}
end

-- returns the next batch
-- a wrapper of getBatch(), which is meant to be overriden by subclasses
-- |return|: a table of two tables, each corresponding to 
-- the batch for dataset A and dataset B
function BaseDataLoader:GetNextBatch()
  self.data_tm:reset()
  self.data_tm:resume()
  local dataA, dataB, pathA, pathB = self:LoadBatchForAllDatasets()
  self.data_tm:stop()
  return dataA, dataB, pathA, pathB
end

function BaseDataLoader:time_elapsed_to_fetch_data()
  return self.data_tm:time().real
end

-- returns the size of each dataset
function BaseDataLoader:size(dataset)
  return 0
end





