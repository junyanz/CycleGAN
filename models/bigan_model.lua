local class = require 'class'
require 'models.base_model'
require 'models.architectures'
require 'util.image_pool'
util = paths.dofile('../util/util.lua')
content = paths.dofile('../util/content_loss.lua')

BiGANModel = class('BiGANModel', 'BaseModel')

function BiGANModel:__init(conf)
  BaseModel.__init(self, conf)
  conf = conf or {}
end

function BiGANModel:model_name()
  return 'BiGANModel'
end

function BiGANModel:InitializeStates(use_wgan)
  optimState = {learningRate=opt.lr, beta1=opt.beta1,}
  return optimState
end
-- Defines models and networks
function BiGANModel:Initialize(opt)
  if opt.test == 0 then
    self.realABPool = ImagePool(opt.pool_size)
    self.fakeABPool = ImagePool(opt.pool_size)
  end
  -- define tensors
  local d_input_nc = opt.input_nc + opt.output_nc
  self.real_AB = torch.Tensor(opt.batchSize, d_input_nc, opt.fineSize, opt.fineSize)
  self.fake_AB = torch.Tensor(opt.batchSize, d_input_nc, opt.fineSize, opt.fineSize)
  -- load/define models
  self.criterionGAN = nn.MSECriterion()

  local netG,  netE, netD = nil, nil, nil
  if opt.continue_train == 1 then
    if opt.test == 1 then -- which_epoch option exists in test mode
      netG = util.load_test_model('G', opt)
      netE = util.load_test_model('E', opt)
      netD = util.load_test_model('D', opt)
    else
      netG = util.load_model('G', opt)
      netE = util.load_model('E', opt)
      netD = util.load_model('D', opt)
    end
  else
    -- netG_test = defineG(opt.input_nc, opt.output_nc, opt.ngf, "resnet_unet", opt.arch)
    -- os.exit()
    netD = defineD(d_input_nc, opt.ndf, opt.which_model_netD, opt.n_layers_D, false)  -- no sigmoid layer
    print('netD...', netD)
    netG = defineG(opt.input_nc, opt.output_nc, opt.ngf, opt.which_model_netG, opt.arch)
    print('netG...', netG)
    netE = defineG(opt.output_nc, opt.input_nc, opt.ngf, opt.which_model_netG, opt.arch)
    print('netE...', netE)

  end

  self.netD = netD
  self.netG = netG
  self.netE = netE

  -- define real/fake labels
  netD_output_size = self.netD:forward(self.real_AB):size()
  self.fake_label = torch.Tensor(netD_output_size):fill(0.0)
  self.real_label = torch.Tensor(netD_output_size):fill(1.0) -- no soft smoothing

  self.optimStateD = self:InitializeStates()
  self.optimStateG = self:InitializeStates()
  self.optimStateE = self:InitializeStates()
  self.A_idx = {{}, {1, opt.input_nc}, {}, {}}
  self.B_idx = {{}, {opt.input_nc+1, opt.input_nc+opt.output_nc}, {}, {}}
  self:RefreshParameters()

  print('---------- # Learnable Parameters --------------')
  print(('G = %d'):format(self.parametersG:size(1)))
  print(('E = %d'):format(self.parametersE:size(1)))
  print(('D = %d'):format(self.parametersD:size(1)))
  print('------------------------------------------------')
  -- os.exit()
end

-- Runs the forward pass of the network and
-- saves the result to member variables of the class
function BiGANModel:Forward(input, opt)
  if opt.which_direction == 'BtoA' then
   	local temp = input.real_A
   	input.real_A = input.real_B
   	input.real_B = temp
   end
   self.real_AB[self.A_idx]:copy(input.real_A)
   self.fake_AB[self.B_idx]:copy(input.real_B)
   self.real_A = self.real_AB[self.A_idx]
   self.real_B = self.fake_AB[self.B_idx]
   self.fake_B = self.netG:forward(self.real_A):clone()
   self.fake_A = self.netE:forward(self.real_B):clone()
   self.real_AB[self.B_idx]:copy(self.fake_B) -- real_AB: real_A, fake_B  -> real_label
   self.fake_AB[self.A_idx]:copy(self.fake_A) -- fake_AB: fake_A, real_B  -> fake_label
  --  if opt.test == 0 then
  --    self.real_AB = self.realABPool:Query(self.real_AB)   -- batch history
  --    self.fake_AB = self.fakeABPool:Query(self.fake_AB)   -- batch history
  --  end
end

-- create closure to evaluate f(X) and df/dX of discriminator
function BiGANModel:fDx_basic(x, gradParams, netD, real_AB, fake_AB, opt)
  util.BiasZero(netD)
  gradParams:zero()
  -- Real  log(D_A(B))
  local output = netD:forward(real_AB):clone()
  local errD_real = self.criterionGAN:forward(output, self.real_label)
  local df_do = self.criterionGAN:backward(output, self.real_label)
  netD:backward(real_AB, df_do)
  -- Fake  + log(1 - D_A(G(A)))
  output = netD:forward(fake_AB):clone()
  local errD_fake = self.criterionGAN:forward(output, self.fake_label)
  local df_do2 = self.criterionGAN:backward(output, self.fake_label)
  netD:backward(fake_AB, df_do2)
  -- Compute loss
  local errD = (errD_real + errD_fake) / 2.0
  return errD, gradParams
end


function BiGANModel:fDx(x, opt)
  -- use image pool that stores the old fake images
  real_AB = self.realABPool:Query(self.real_AB)
  fake_AB = self.fakeABPool:Query(self.fake_AB)
  self.errD, gradParams = self:fDx_basic(x, self.gradParametersD, self.netD, real_AB, fake_AB, opt)
  return self.errD, gradParams
end



function BiGANModel:fGx_basic(x, netG, netD, gradParametersG, opt)
  util.BiasZero(netG)
  util.BiasZero(netD)
  gradParametersG:zero()

  -- First. G(A) should fake the discriminator
  local output = netD:forward(self.real_AB):clone()
  local errG = self.criterionGAN:forward(output, self.fake_label)
  local dgan_loss_dd = self.criterionGAN:backward(output, self.fake_label)
  local dgan_loss_do = netD:updateGradInput(self.real_AB, dgan_loss_dd)
  netG:backward(self.real_A, dgan_loss_do[self.B_idx]) -- real_AB: real_A, fake_B  -> real_label
  return gradParametersG, errG
end


function BiGANModel:fGx(x, opt)
  self.gradParametersG, self.errG =  self:fGx_basic(x, self.netG, self.netD,
             self.gradParametersG, opt)
  return self.errG, self.gradParametersG
end


function BiGANModel:fEx_basic(x, netE, netD, gradParametersE, opt)
  util.BiasZero(netE)
  util.BiasZero(netD)
  gradParametersE:zero()

  -- First. G(A) should fake the discriminator
  local output = netD:forward(self.fake_AB):clone()
  local errE= self.criterionGAN:forward(output, self.real_label)
  local dgan_loss_dd = self.criterionGAN:backward(output, self.real_label)
  local dgan_loss_do = netD:updateGradInput(self.fake_AB, dgan_loss_dd)
  netE:backward(self.real_B, dgan_loss_do[self.A_idx])-- fake_AB: fake_A, real_B  -> fake_label
  return gradParametersE, errE
end


function BiGANModel:fEx(x, opt)
  self.gradParametersE, self.errE =  self:fEx_basic(x, self.netE, self.netD,
             self.gradParametersE, opt)
  return self.errE, self.gradParametersE
end


function BiGANModel:OptimizeParameters(opt)
  local fG = function(x) return self:fGx(x, opt) end
  local fE = function(x) return self:fEx(x, opt) end
  local fD = function(x) return self:fDx(x, opt) end
  optim.adam(fD, self.parametersD, self.optimStateD)
  optim.adam(fG, self.parametersG, self.optimStateG)
  optim.adam(fE, self.parametersE, self.optimStateE)
end

function BiGANModel:RefreshParameters()
  self.parametersD, self.gradParametersD = nil, nil -- nil them to avoid spiking memory
  self.parametersG, self.gradParametersG = nil, nil
  self.parametersE, self.gradParametersE = nil, nil
  -- define parameters of optimization
  self.parametersD, self.gradParametersD = self.netD:getParameters()
  self.parametersG, self.gradParametersG = self.netG:getParameters()
  self.parametersE, self.gradParametersE = self.netE:getParameters()
end

function BiGANModel:Save(prefix, opt)
  util.save_model(self.netG, prefix .. '_net_G.t7', 1)
  util.save_model(self.netE, prefix .. '_net_E.t7', 1)
  util.save_model(self.netD, prefix .. '_net_D.t7', 1)
end

function BiGANModel:GetCurrentErrorDescription()
  description = ('D: %.4f  G: %.4f  E: %.4f'):format(
                         self.errD and self.errD or -1,
                         self.errG and self.errG or -1,
                         self.errE and self.errE or -1)
  return description
end

function BiGANModel:GetCurrentErrors()
  local errors = {errD=self.errD, errG=self.errG, errE=self.errE}
  return errors
end

-- returns a string that describes the display plot configuration
function BiGANModel:DisplayPlot(opt)
  return 'errD,errG,errE'
end
function BiGANModel:UpdateLearningRate(opt)
  local lrd = opt.lr / opt.niter_decay
  local old_lr = self.optimStateD['learningRate']
  local lr =  old_lr - lrd
  self.optimStateD['learningRate'] = lr
  self.optimStateG['learningRate'] = lr
  self.optimStateE['learningRate'] = lr
  print(('update learning rate: %f -> %f'):format(old_lr, lr))
end

local function MakeIm3(im)
  -- print('before im_size', im:size())
  local im3 = nil
  if im:size(2) == 1 then
    im3 = torch.repeatTensor(im, 1,3,1,1)
  else
    im3 = im
  end
  -- print('after im_size', im:size())
  -- print('after im3_size', im3:size())
  return im3
end
function BiGANModel:GetCurrentVisuals(opt, size)
  if not size then
    size = opt.display_winsize
  end

  local visuals = {}
  table.insert(visuals, {img=MakeIm3(self.real_A), label='real_A'})
  table.insert(visuals, {img=MakeIm3(self.fake_B), label='fake_B'})
  table.insert(visuals, {img=MakeIm3(self.real_B), label='real_B'})
  table.insert(visuals, {img=MakeIm3(self.fake_A), label='fake_A'})
  return visuals
end
