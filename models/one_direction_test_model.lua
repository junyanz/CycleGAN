local class = require 'class'
require 'models.base_model'
require 'models.architectures'
require 'util.image_pool'

util = paths.dofile('../util/util.lua')
OneDirectionTestModel = class('OneDirectionTestModel', 'BaseModel')

function OneDirectionTestModel:__init(conf)
  BaseModel.__init(self, conf)
  conf = conf or {}
end

function OneDirectionTestModel:model_name()
  return 'OneDirectionTestModel'
end

-- Defines models and networks
function OneDirectionTestModel:Initialize(opt)
  -- define tensors
  self.real_A = torch.Tensor(opt.batchSize, opt.input_nc, opt.fineSize, opt.fineSize)

  -- load/define models
  self.netG_A = util.load_test_model('G', opt)

  -- setup optnet to save a bit of memory
  if opt.use_optnet == 1 then
    local optnet = require 'optnet'
    local sample_input = torch.randn(1, opt.input_nc, 2, 2)
    optnet.optimizeMemory(self.netG_A, sample_input, {inplace=true, reuseBuffers=true})
  end

  self:RefreshParameters()

  print('---------- # Learnable Parameters --------------')
  print(('G_A = %d'):format(self.parametersG_A:size(1)))
  print('------------------------------------------------')
end

-- Runs the forward pass of the network and
-- saves the result to member variables of the class
function OneDirectionTestModel:Forward(input, opt)
  if opt.which_direction == 'BtoA' then
  	input.real_A = input.real_B:clone()
  end

  self.real_A = input.real_A:clone()
  if opt.gpu > 0 then
    self.real_A = self.real_A:cuda()
  end

  self.fake_B = self.netG_A:forward(self.real_A):clone()
end

function OneDirectionTestModel:RefreshParameters()
  self.parametersG_A, self.gradparametersG_A = nil, nil
  self.parametersG_A, self.gradparametersG_A = self.netG_A:getParameters()
end


local function MakeIm3(im)
  if im:size(2) == 1 then
    local im3 = torch.repeatTensor(im, 1,3,1,1)
    return im3
  else
    return im
  end
end

function OneDirectionTestModel:GetCurrentVisuals(opt, size)
  if not size then
    size = opt.display_winsize
  end

  local visuals = {}
  table.insert(visuals, {img=MakeIm3(self.real_A), label='real_A'})
  table.insert(visuals, {img=MakeIm3(self.fake_B), label='fake_B'})
  return visuals
end
