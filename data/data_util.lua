local data_util = {}

require 'torch'
-- options =  require '../options.lua'
-- load dataset from the file system
-- |name|: name of the dataset. It's currently either 'A' or 'B'
function data_util.load_dataset(name, opt, nc)
  local tensortype = torch.getdefaulttensortype()
  torch.setdefaulttensortype('torch.FloatTensor')

  local new_opt = options.clone(opt)
  new_opt.manualSeed = torch.random(1, 10000) -- fix seed
  new_opt.nc = nc
  torch.manualSeed(new_opt.manualSeed)
  local data_loader = paths.dofile('../data/data.lua')
  new_opt.phase = new_opt.phase .. name
  local data = data_loader.new(new_opt.nThreads, new_opt)
  print("Dataset Size " .. name .. ": ", data:size())

  torch.setdefaulttensortype(tensortype)
  return data
end

return data_util
