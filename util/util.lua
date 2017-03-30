--
-- code derived from https://github.com/soumith/dcgan.torch
--

local util = {}

require 'torch'


function util.BiasZero(net)
  net:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)
end


function util.checkEqual(A, B, name)
  local dif = (A:float()-B:float()):abs():mean()
  print(name, dif)
end

function util.containsValue(table, value)
  for k, v in pairs(table) do
    if v == value then return true end
  end
  return false
end


function util.CheckTensor(A, name)
  print(name, A:min(), A:max(), A:mean())
end


function util.normalize(img)
  -- rescale image to 0 .. 1
  local min = img:min()
  local max = img:max()

  img = torch.FloatTensor(img:size()):copy(img)
  img:add(-min):mul(1/(max-min))
  return img
end

function util.normalizeBatch(batch)
	for i = 1, batch:size(1) do
		batch[i] = util.normalize(batch[i]:squeeze())
	end
	return batch
end

function util.basename_batch(batch)
	for i = 1, #batch do
		batch[i] = paths.basename(batch[i])
	end
	return batch
end



-- default preprocessing
--
-- Preprocesses an image before passing it to a net
-- Converts from RGB to BGR and rescales from [0,1] to [-1,1]
function util.preprocess(img)
    -- RGB to BGR
    if img:size(1) == 3 then
      local perm = torch.LongTensor{3, 2, 1}
      img = img:index(1, perm)
    end
    -- [0,1] to [-1,1]
    img = img:mul(2):add(-1)

    -- check that input is in expected range
    assert(img:max()<=1,"badly scaled inputs")
    assert(img:min()>=-1,"badly scaled inputs")

    return img
end

-- Undo the above preprocessing.
function util.deprocess(img)
    -- BGR to RGB
    if img:size(1) == 3 then
      local perm = torch.LongTensor{3, 2, 1}
      img = img:index(1, perm)
    end

    -- [-1,1] to [0,1]
    img = img:add(1):div(2)

    return img
end

function util.preprocess_batch(batch)
	for i = 1, batch:size(1) do
		batch[i] = util.preprocess(batch[i]:squeeze())
	end
	return batch
end

function util.print_tensor(name, x)
  print(name, x:size(), x:min(), x:mean(), x:max())
end

function util.deprocess_batch(batch)
	for i = 1, batch:size(1) do
		batch[i] = util.deprocess(batch[i]:squeeze())
	end
	return batch
end


function util.scaleBatch(batch,s1,s2)
  -- print('s1', s1)
  -- print('s2', s2)
	local scaled_batch = torch.Tensor(batch:size(1),batch:size(2),s1,s2)
	for i = 1, batch:size(1) do
		scaled_batch[i] = image.scale(batch[i],s1,s2):squeeze()
	end
	return scaled_batch
end



function util.toTrivialBatch(input)
  return input:reshape(1,input:size(1),input:size(2),input:size(3))
end
function util.fromTrivialBatch(input)
    return input[1]
end

-- input is between -1 and 1
function util.jitter(input)
  local noise = torch.rand(input:size())/256.0
  input:add(1.0):mul(0.5*255.0/256.0):add(noise):add(-0.5):mul(2.0)
  --local scaled = (input+1.0)*0.5
  --local jittered = scaled*255.0/256.0 + torch.rand(input:size())/256.0
  --local scaled_back = (jittered-0.5)*2.0
  --return scaled_back
end

function util.scaleImage(input, loadSize)

    -- replicate bw images to 3 channels
    if input:size(1)==1 then
    	input = torch.repeatTensor(input,3,1,1)
    end

    input = image.scale(input, loadSize, loadSize)

    return input
end

function util.getAspectRatio(path)
	local input = image.load(path, 3, 'float')
	local ar = input:size(3)/input:size(2)
	return ar
end

function util.loadImage(path, loadSize, nc)
  local input = image.load(path, 3, 'float')
  input= util.preprocess(util.scaleImage(input, loadSize))

  if nc == 1 then
    input = input[{{1}, {}, {}}]
  end

  return input
end

function file_exists(filename)
   local f = io.open(filename,"r")
   if f ~= nil then io.close(f) return true else return false end
end

-- TO DO: loading code is rather hacky; clean it up and make sure it works on all types of nets / cpu/gpu configurations
function load_helper(filename, opt)
  fileExists = file_exists(filename)
  if not fileExists then
    print('model not found!    ' .. filename)
    return nil
  end
  print(('loading previously trained model (%s)'):format(filename))
	if opt.norm == 'instance' then
	  print('use InstanceNormalization')
	  require 'util.InstanceNormalization'
	end

	if opt.cudnn>0 then
		require 'cudnn'
	end

	local net = torch.load(filename)
	if opt.gpu > 0 then
		require 'cunn'
		net:cuda()

		-- calling cuda on cudnn saved nngraphs doesn't change all variables to cuda, so do it below
		if net.forwardnodes then
			for i=1,#net.forwardnodes do
				if net.forwardnodes[i].data.module then
					net.forwardnodes[i].data.module:cuda()
				end
			end
		end

	else
		net:float()
	end
	net:apply(function(m) if m.weight then
	    m.gradWeight = m.weight:clone():zero();
	    m.gradBias = m.bias:clone():zero(); end end)
	return net
end

function util.load_model(name, opt)
  -- if opt['lambda_'.. name] > 0.0 then
  -- print('not loading model '.. opt.checkpoints_dir .. opt.name ..
  --         'latest_net_' .. name .. '.t7' .. ' because opt.lambda is not greater than zero')
  return load_helper(paths.concat(opt.checkpoints_dir, opt.name,
                                'latest_net_' .. name .. '.t7'), opt)
  -- end
end

function util.load_test_model(name, opt)
  return load_helper(paths.concat(opt.checkpoints_dir, opt.name,
                                    opt.which_epoch .. '_net_' .. name .. '.t7'), opt)
end


-- load dataset from the file system
-- |name|: name of the dataset. It's currently either 'A' or 'B'
-- function util.load_dataset(name, nc, opt, nc)
--   local tensortype = torch.getdefaulttensortype()
--   torch.setdefaulttensortype('torch.FloatTensor')
--
--   local new_opt = options.clone(opt)
--   new_opt.manualSeed = torch.random(1, 10000) -- fix seed
--   new_opt.nc = nc
--   torch.manualSeed(new_opt.manualSeed)
--   local data_loader = paths.dofile('../data/data.lua')
--   new_opt.phase = new_opt.phase .. name
--   local data = data_loader.new(new_opt.nThreads, new_opt)
--   print("Dataset Size " .. name .. ": ", data:size())
--
--   torch.setdefaulttensortype(tensortype)
--   return data
-- end



function util.cudnn(net)
	require 'cudnn'
	require 'util/cudnn_convert_custom'
	return cudnn_convert_custom(net, cudnn)
end

function util.save_model(net, net_name, weight)
  if weight > 0.0 then
	   torch.save(paths.concat(opt.checkpoints_dir, opt.name, net_name), net:clearState())
  end
end




return util
