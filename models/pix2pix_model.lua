local class = require 'class'
require 'models.base_model'
require 'models.architectures'
require 'util.image_pool'
util = paths.dofile('../util/util.lua')
Pix2PixModel = class('Pix2PixModel', 'BaseModel')

function Pix2PixModel:__init(conf)
   conf = conf or {}
end

-- Returns the name of the model
function Pix2PixModel:model_name()
   return 'Pix2PixModel'
end

function Pix2PixModel:InitializeStates()
  return {learningRate=opt.lr, beta1=opt.beta1,}
end

-- Defines models and networks
function Pix2PixModel:Initialize(opt)  -- use lsgan
  -- define tensors
  local d_input_nc = opt.input_nc + opt.output_nc
  self.real_AB = torch.Tensor(opt.batchSize, d_input_nc, opt.fineSize, opt.fineSize)
  self.fake_AB = torch.Tensor(opt.batchSize, d_input_nc, opt.fineSize, opt.fineSize)
  if opt.test == 0 then
    self.fakeABPool = ImagePool(opt.pool_size)
  end
  -- load/define models
  self.criterionGAN = nn.MSECriterion()
  self.criterionL1 = nn.AbsCriterion()

  local netG, netD = nil, nil
  if opt.continue_train == 1 then
    if opt.test == 1 then -- only load model G for test
      netG = util.load_test_model('G', opt)
    else
      netG = util.load_model('G', opt)
      netD = util.load_model('D', opt)
    end
  else
    netG = defineG(opt.input_nc, opt.output_nc, opt.ngf, opt.which_model_netG)
    netD = defineD(d_input_nc, opt.ndf, opt.which_model_netD, opt.n_layers_D, false)  -- with sigmoid
  end

  self.netD = netD
  self.netG = netG

  -- define real/fake labels
  if opt.test == 0 then
    netD_output_size = self.netD:forward(self.real_AB):size()
    self.fake_label = torch.Tensor(netD_output_size):fill(0.0)
    self.real_label = torch.Tensor(netD_output_size):fill(1.0) -- no soft smoothing

    self.optimStateD = self:InitializeStates()
    self.optimStateG = self:InitializeStates()

    self:RefreshParameters()

    print('---------- # Learnable Parameters --------------')
    print(('G = %d'):format(self.parametersG:size(1)))
    print(('D = %d'):format(self.parametersD:size(1)))
    print('------------------------------------------------')
  end

  self.A_idx = {{}, {1, opt.input_nc}, {}, {}}
  self.B_idx = {{}, {opt.input_nc+1, opt.input_nc+opt.output_nc}, {}, {}}
end

-- Runs the forward pass of the network
function Pix2PixModel:Forward(input, opt)
  if opt.which_direction == 'BtoA' then
  	local temp = input.real_A
  	input.real_A = input.real_B
  	input.real_B = temp
  end

  if opt.test == 0 then
    self.real_AB[self.A_idx]:copy(input.real_A)
    self.real_AB[self.B_idx]:copy(input.real_B)
    self.real_A = self.real_AB[self.A_idx]
    self.real_B = self.real_AB[self.B_idx]

    self.fake_AB[self.A_idx]:copy(self.real_A)
    self.fake_B = self.netG:forward(self.real_A):clone()
    self.fake_AB[self.B_idx]:copy(self.fake_B)
  else
    if opt.gpu > 0 then
      self.real_A = input.real_A:cuda()
      self.real_B = input.real_B:cuda()
    else
      self.real_A = input.real_A:clone()
      self.real_B = input.real_B:clone()
    end
    self.fake_B = self.netG:forward(self.real_A):clone()
  end
end

-- create closure to evaluate f(X) and df/dX of discriminator
function Pix2PixModel:fDx_basic(x, gradParams, netD, netG, real, fake, opt)
  util.BiasZero(netD)
  util.BiasZero(netG)
  gradParams:zero()

  -- Real  log(D(B))
  local output = netD:forward(real)
  local errD_real = self.criterionGAN:forward(output, self.real_label)
  local df_do = self.criterionGAN:backward(output, self.real_label)
  netD:backward(real, df_do)
  -- Fake  + log(1 - D(G(A)))
  output = netD:forward(fake)
  local errD_fake = self.criterionGAN:forward(output, self.fake_label)
  local df_do2 = self.criterionGAN:backward(output, self.fake_label)
  netD:backward(fake, df_do2)
  -- calculate loss
  local errD = (errD_real + errD_fake) / 2.0
  return errD, gradParams
end


function Pix2PixModel:fDx(x, opt)
  fake_AB = self.fakeABPool:Query(self.fake_AB)
  self.errD, gradParams = self:fDx_basic(x, self.gradParametersD, self.netD, self.netG,
                                     self.real_AB, fake_AB, opt)
  return self.errD, gradParams
end

function Pix2PixModel:fGx_basic(x, netG, netD, real, fake, gradParametersG, opt)
  util.BiasZero(netG)
  util.BiasZero(netD)
  gradParametersG:zero()

  -- First. G(A) should fake the discriminator
  local output = netD:forward(fake)
  local errG = self.criterionGAN:forward(output, self.real_label)
  local dgan_loss_dd = self.criterionGAN:backward(output, self.real_label)
  local dgan_loss_do = netD:updateGradInput(fake, dgan_loss_dd)

  -- Second. G(A) should be close to the real
  real_B = real[self.B_idx]
  real_A = real[self.A_idx]
  fake_B = fake[self.B_idx]
  local errL1 = self.criterionL1:forward(fake_B, real_B) * opt.lambda_A
  local dl1_loss_do = self.criterionL1:backward(fake_B, real_B) * opt.lambda_A
  netG:backward(real_A, dgan_loss_do[self.B_idx] + dl1_loss_do)

  return gradParametersG, errG, errL1
end

function Pix2PixModel:fGx(x, opt)
  self.gradParametersG, self.errG, self.errL1 =  self:fGx_basic(x, self.netG, self.netD,
             self.real_AB, self.fake_AB, self.gradParametersG, opt)
  return self.errG, self.gradParametersG
end

-- Runs the backprop gradient descent
-- Corresponds to a single batch of data
function Pix2PixModel:OptimizeParameters(opt)
  local fD = function(x) return self:fDx(x, opt) end
  local fG = function(x) return self:fGx(x, opt) end
  optim.adam(fD, self.parametersD, self.optimStateD)
  optim.adam(fG, self.parametersG, self.optimStateG)
end

-- This function can be used to reset momentum after each epoch
function Pix2PixModel:RefreshParameters()
  self.parametersD, self.gradParametersD = nil, nil -- nil them to avoid spiking memory
  self.parametersG, self.gradParametersG = nil, nil

  -- define parameters of optimization
  self.parametersG, self.gradParametersG = self.netG:getParameters()
  self.parametersD, self.gradParametersD = self.netD:getParameters()
end

-- This function updates the learning rate; lr for the first opt.niter iterations; graduatlly decreases the lr to 0 for the next opt.niter_decay iterations
function Pix2PixModel:UpdateLearningRate(opt)
  local lrd = opt.lr / opt.niter_decay
  local old_lr = self.optimStateD['learningRate']
  local lr =  old_lr - lrd
  self.optimStateD['learningRate'] = lr
  self.optimStateG['learningRate'] = lr
  print(('update learning rate: %f -> %f'):format(old_lr, lr))
end


-- Save the current model to the file system
function Pix2PixModel:Save(prefix, opt)
  util.save_model(self.netG, prefix .. '_net_G.t7', 1.0)
  util.save_model(self.netD, prefix .. '_net_D.t7', 1.0)
end

-- returns a string that describes the current errors
function Pix2PixModel:GetCurrentErrorDescription()
  description = ('G: %.4f  D: %.4f L1: %.4f'):format(
      self.errG and self.errG or -1, self.errD and self.errD or -1, self.errL1 and self.errL1 or -1)
  return description

end


-- returns a string that describes the display plot configuration
function Pix2PixModel:DisplayPlot(opt)
  return 'errG,errD,errL1'
end


-- returns current errors
function Pix2PixModel:GetCurrentErrors()
  local errors = {errG=self.errG, errD=self.errD, errL1=self.errL1}
  return errors
end

-- returns a table of image/label pairs that describe
-- the current results.
-- |return|: a table of table. List of image/label pairs
function Pix2PixModel:GetCurrentVisuals(opt, size)
  if not size then
    size = opt.display_winsize
  end

  local visuals = {}
  table.insert(visuals, {img=self.real_A, label='real_A'})
  table.insert(visuals, {img=self.fake_B, label='fake_B'})
  table.insert(visuals, {img=self.real_B, label='real_B'})

  return visuals
end
