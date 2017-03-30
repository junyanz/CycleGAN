--------------------------------------------------------------------------------
-- Base Class for Providing Models
--------------------------------------------------------------------------------

local class = require 'class'

BaseModel = class('BaseModel')

function BaseModel:__init(conf)
   conf = conf or {}
end

-- Returns the name of the model
function BaseModel:model_name()
   return 'DoesNothingModel'
end

-- Defines models and networks
function BaseModel:Initialize(opt)
  models = {}
  return models
end

-- Runs the forward pass of the network
function BaseModel:Forward(input, opt)
  output = {}
  return output
end

-- Runs the backprop gradient descent
-- Corresponds to a single batch of data
function BaseModel:OptimizeParameters(opt)
end

-- This function can be used to reset momentum after each epoch
function BaseModel:RefreshParameters(opt)
end

-- This function can be used to reset momentum after each epoch
function BaseModel:UpdateLearningRate(opt)
end
-- Save the current model to the file system
function BaseModel:Save(prefix, opt)
end

-- returns a string that describes the current errors
function BaseModel:GetCurrentErrorDescription()
  return "No Error exists in BaseModel"
end

-- returns current errors
function BaseModel:GetCurrentErrors(opt)
  return {}
end

-- returns a table of image/label pairs that describe
-- the current results.
-- |return|: a table of table. List of image/label pairs
function BaseModel:GetCurrentVisuals(opt, size)
  return {}
end

-- returns a string that describes the display plot configuration 
function BaseModel:DisplayPlot(opt)
  return {}
end
