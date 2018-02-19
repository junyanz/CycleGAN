local class = require 'class'
require 'models.base_model'
require 'models.architectures'
require 'util.image_pool'

util = paths.dofile('../util/util.lua')
CycleGANModel = class('CycleGANModel', 'BaseModel')

function CycleGANModel:__init(conf)
  BaseModel.__init(self, conf)
  conf = conf or {}
end

function CycleGANModel:model_name()
  return 'CycleGANModel'
end

function CycleGANModel:InitializeStates(use_wgan)
  optimState = {learningRate=opt.lr, beta1=opt.beta1,}
  return optimState
end
-- Defines models and networks
function CycleGANModel:Initialize(opt)
  if opt.test == 0 then
    self.fakeAPool = ImagePool(opt.pool_size)
    self.fakeBPool = ImagePool(opt.pool_size)
  end
  -- define tensors
  if opt.test == 0 then  -- allocate tensors for training
    self.real_A = torch.Tensor(opt.batchSize, opt.input_nc, opt.fineSize, opt.fineSize)
    self.real_B = torch.Tensor(opt.batchSize, opt.output_nc, opt.fineSize, opt.fineSize)
    self.fake_A = torch.Tensor(opt.batchSize, opt.input_nc, opt.fineSize, opt.fineSize)
    self.fake_B = torch.Tensor(opt.batchSize, opt.output_nc, opt.fineSize, opt.fineSize)
    self.rec_A  = torch.Tensor(opt.batchSize, opt.input_nc, opt.fineSize, opt.fineSize)
    self.rec_B  = torch.Tensor(opt.batchSize, opt.output_nc, opt.fineSize, opt.fineSize)
  end
  -- load/define models
  local use_lsgan = ((opt.use_lsgan ~= nil) and (opt.use_lsgan == 1))
  if not use_lsgan then
    self.criterionGAN = nn.BCECriterion()
  else
    self.criterionGAN = nn.MSECriterion()
  end
  self.criterionRec = nn.AbsCriterion()

  local netG_A, netD_A, netG_B, netD_B = nil, nil, nil, nil
  if opt.continue_train == 1 then
    if opt.test == 1 then -- test mode
      netG_A = util.load_test_model('G_A', opt)
      netG_B = util.load_test_model('G_B', opt)

      --setup optnet to save a little bit of memory
      if opt.use_optnet == 1 then
        local sample_input = torch.randn(1, opt.input_nc, 2, 2)
        local optnet = require 'optnet'
        optnet.optimizeMemory(netG_A, sample_input, {inplace=true, reuseBuffers=true})
        optnet.optimizeMemory(netG_B, sample_input, {inplace=true, reuseBuffers=true})
      end
    else
      netG_A = util.load_model('G_A', opt)
      netG_B = util.load_model('G_B', opt)
      netD_A = util.load_model('D_A', opt)
      netD_B = util.load_model('D_B', opt)
    end
  else
    local use_sigmoid = (not use_lsgan)
    -- netG_test = defineG(opt.input_nc, opt.output_nc, opt.ngf, "resnet_unet", opt.arch)
    -- os.exit()
    netG_A = defineG(opt.input_nc, opt.output_nc, opt.ngf, opt.which_model_netG, opt.arch)
    print('netG_A...', netG_A)
    netD_A = defineD(opt.output_nc, opt.ndf, opt.which_model_netD, opt.n_layers_D, use_sigmoid)  -- no sigmoid layer
    print('netD_A...', netD_A)
    netG_B = defineG(opt.output_nc, opt.input_nc, opt.ngf, opt.which_model_netG, opt.arch)
    print('netG_B...', netG_B)
    netD_B = defineD(opt.input_nc, opt.ndf, opt.which_model_netD, opt.n_layers_D, use_sigmoid)  -- no sigmoid layer
    print('netD_B', netD_B)
  end

  self.netD_A = netD_A
  self.netG_A = netG_A
  self.netG_B = netG_B
  self.netD_B = netD_B

  -- define real/fake labels
  if opt.test == 0 then
    local D_A_size = self.netD_A:forward(self.real_B):size()  -- hack: assume D_size_A = D_size_B
    self.fake_label_A = torch.Tensor(D_A_size):fill(0.0)
    self.real_label_A = torch.Tensor(D_A_size):fill(1.0) -- no soft smoothing
    local D_B_size = self.netD_B:forward(self.real_A):size()  -- hack: assume D_size_A = D_size_B
    self.fake_label_B = torch.Tensor(D_B_size):fill(0.0)
    self.real_label_B = torch.Tensor(D_B_size):fill(1.0) -- no soft smoothing
    self.optimStateD_A = self:InitializeStates()
    self.optimStateG_A = self:InitializeStates()
    self.optimStateD_B = self:InitializeStates()
    self.optimStateG_B = self:InitializeStates()
    self:RefreshParameters()
    print('---------- # Learnable Parameters --------------')
    print(('G_A = %d'):format(self.parametersG_A:size(1)))
    print(('D_A = %d'):format(self.parametersD_A:size(1)))
    print(('G_B = %d'):format(self.parametersG_B:size(1)))
    print(('D_B = %d'):format(self.parametersD_B:size(1)))
    print('------------------------------------------------')
  end
end

-- Runs the forward pass of the network and
-- saves the result to member variables of the class
function CycleGANModel:Forward(input, opt)
  if opt.which_direction == 'BtoA' then
  	local temp = input.real_A:clone()
  	input.real_A = input.real_B:clone()
  	input.real_B = temp
  end

  if opt.test == 0 then
    self.real_A:copy(input.real_A)
    self.real_B:copy(input.real_B)
  end

  if opt.test == 1 then  -- forward for test
    if opt.gpu > 0 then
      self.real_A = input.real_A:cuda()
      self.real_B = input.real_B:cuda()
    else
      self.real_A = input.real_A:clone()
      self.real_B = input.real_B:clone()
    end
    self.fake_B = self.netG_A:forward(self.real_A):clone()
    self.fake_A = self.netG_B:forward(self.real_B):clone()
    self.rec_A  = self.netG_B:forward(self.fake_B):clone()
    self.rec_B  = self.netG_A:forward(self.fake_A):clone()
  end
end

-- create closure to evaluate f(X) and df/dX of discriminator
function CycleGANModel:fDx_basic(x, gradParams, netD, netG, real, fake, real_label, fake_label, opt)
  util.BiasZero(netD)
  util.BiasZero(netG)
  gradParams:zero()
  -- Real  log(D_A(B))
  local output = netD:forward(real)
  local errD_real = self.criterionGAN:forward(output, real_label)
  local df_do = self.criterionGAN:backward(output, real_label)
  netD:backward(real, df_do)
  -- Fake  + log(1 - D_A(G_A(A)))
  output = netD:forward(fake)
  local errD_fake = self.criterionGAN:forward(output, fake_label)
  local df_do2 = self.criterionGAN:backward(output, fake_label)
  netD:backward(fake, df_do2)
  -- Compute loss
  local errD = (errD_real + errD_fake) / 2.0
  return errD, gradParams
end


function CycleGANModel:fDAx(x, opt)
  -- use image pool that stores the old fake images
  fake_B = self.fakeBPool:Query(self.fake_B)
  self.errD_A, gradParams = self:fDx_basic(x, self.gradparametersD_A, self.netD_A, self.netG_A,
                            self.real_B, fake_B, self.real_label_A, self.fake_label_A, opt)
  return self.errD_A, gradParams
end


function CycleGANModel:fDBx(x, opt)
  -- use image pool that stores the old fake images
  fake_A = self.fakeAPool:Query(self.fake_A)
  self.errD_B, gradParams = self:fDx_basic(x, self.gradparametersD_B, self.netD_B, self.netG_B,
                            self.real_A, fake_A, self.real_label_B, self.fake_label_B, opt)
  return self.errD_B, gradParams
end


function CycleGANModel:fGx_basic(x, gradParams, netG, netD, netE, real, real2, real_label, lambda1, lambda2, opt)
  util.BiasZero(netD)
  util.BiasZero(netG)
  util.BiasZero(netE)  -- inverse mapping
  gradParams:zero()

  -- G should be identity if real2 is fed.
  local errI = nil
  local identity = nil
  if opt.lambda_identity > 0 then
    identity = netG:forward(real2):clone()
    errI = self.criterionRec:forward(identity, real2) * lambda2 * opt.lambda_identity
    local didentity_loss_do = self.criterionRec:backward(identity, real2):mul(lambda2):mul(opt.lambda_identity)
    netG:backward(real2, didentity_loss_do)
  end

  --- GAN loss: D_A(G_A(A))
  local fake = netG:forward(real):clone()
  local output = netD:forward(fake)
  local errG = self.criterionGAN:forward(output, real_label)
  local df_do1 = self.criterionGAN:backward(output, real_label)
  local df_d_GAN = netD:updateGradInput(fake, df_do1) --

  -- forward cycle loss
  local rec = netE:forward(fake):clone()
  local errRec = self.criterionRec:forward(rec, real) * lambda1
  local df_do2 = self.criterionRec:backward(rec, real):mul(lambda1)
  local df_do_rec = netE:updateGradInput(fake, df_do2)

  netG:backward(real, df_d_GAN + df_do_rec)

  -- backward cycle loss
  local fake2 = netE:forward(real2)--:clone()
  local rec2 = netG:forward(fake2)--:clone()
  local errAdapt = self.criterionRec:forward(rec2, real2) * lambda2
  local df_do_coadapt = self.criterionRec:backward(rec2, real2):mul(lambda2)
  netG:backward(fake2, df_do_coadapt)

  return gradParams, errG, errRec, errI, fake, rec, identity
end

function CycleGANModel:fGAx(x, opt)
  self.gradparametersG_A, self.errG_A, self.errRec_A, self.errI_A, self.fake_B, self.rec_A, self.identity_B =
  self:fGx_basic(x, self.gradparametersG_A, self.netG_A, self.netD_A, self.netG_B, self.real_A, self.real_B,
  self.real_label_A, opt.lambda_A, opt.lambda_B, opt)
  return self.errG_A, self.gradparametersG_A
end

function CycleGANModel:fGBx(x, opt)
  self.gradparametersG_B, self.errG_B, self.errRec_B, self.errI_B, self.fake_A, self.rec_B, self.identity_A =
  self:fGx_basic(x, self.gradparametersG_B, self.netG_B, self.netD_B, self.netG_A, self.real_B, self.real_A,
  self.real_label_B, opt.lambda_B, opt.lambda_A, opt)
  return self.errG_B, self.gradparametersG_B
end


function CycleGANModel:OptimizeParameters(opt)
  local fDA = function(x) return self:fDAx(x, opt) end
  local fGA = function(x) return self:fGAx(x, opt) end
  local fDB = function(x) return self:fDBx(x, opt) end
  local fGB = function(x) return self:fGBx(x, opt) end

  optim.adam(fGA, self.parametersG_A, self.optimStateG_A)
  optim.adam(fDA, self.parametersD_A, self.optimStateD_A)
  optim.adam(fGB, self.parametersG_B, self.optimStateG_B)
  optim.adam(fDB, self.parametersD_B, self.optimStateD_B)
end

function CycleGANModel:RefreshParameters()
  self.parametersD_A, self.gradparametersD_A = nil, nil -- nil them to avoid spiking memory
  self.parametersG_A, self.gradparametersG_A = nil, nil
  self.parametersG_B, self.gradparametersG_B = nil, nil
  self.parametersD_B, self.gradparametersD_B = nil, nil
  -- define parameters of optimization
  self.parametersG_A, self.gradparametersG_A = self.netG_A:getParameters()
  self.parametersD_A, self.gradparametersD_A = self.netD_A:getParameters()
  self.parametersG_B, self.gradparametersG_B = self.netG_B:getParameters()
  self.parametersD_B, self.gradparametersD_B = self.netD_B:getParameters()
end

function CycleGANModel:Save(prefix, opt)
  util.save_model(self.netG_A, prefix .. '_net_G_A.t7', 1)
  util.save_model(self.netD_A, prefix .. '_net_D_A.t7', 1)
  util.save_model(self.netG_B, prefix .. '_net_G_B.t7', 1)
  util.save_model(self.netD_B, prefix .. '_net_D_B.t7', 1)
end

function CycleGANModel:GetCurrentErrorDescription()
  description = ('[A] G: %.4f  D: %.4f  Rec: %.4f I: %.4f || [B] G: %.4f D: %.4f Rec: %.4f I:%.4f'):format(
                         self.errG_A and self.errG_A or -1,
                         self.errD_A and self.errD_A or -1,
                         self.errRec_A and self.errRec_A or -1,
                         self.errI_A and self.errI_A or -1,
                         self.errG_B and self.errG_B or -1,
                         self.errD_B and self.errD_B or -1,
                         self.errRec_B and self.errRec_B or -1,
                         self.errI_B and self.errI_B or -1)
  return description
end

function CycleGANModel:GetCurrentErrors()
  local errors = {errG_A=self.errG_A, errD_A=self.errD_A, errRec_A=self.errRec_A, errI_A=self.errI_A,
                  errG_B=self.errG_B, errD_B=self.errD_B, errRec_B=self.errRec_B, errI_B=self.errI_B}
  return errors
end

-- returns a string that describes the display plot configuration
function CycleGANModel:DisplayPlot(opt)
  if opt.lambda_identity > 0 then
    return 'errG_A,errD_A,errRec_A,errI_A,errG_B,errD_B,errRec_B,errI_B'
  else
    return 'errG_A,errD_A,errRec_A,errG_B,errD_B,errRec_B'
  end
end

function CycleGANModel:UpdateLearningRate(opt)
  local lrd = opt.lr / opt.niter_decay
  local old_lr = self.optimStateD_A['learningRate']
  local lr =  old_lr - lrd
  self.optimStateD_A['learningRate'] = lr
  self.optimStateD_B['learningRate'] = lr
  self.optimStateG_A['learningRate'] = lr
  self.optimStateG_B['learningRate'] = lr
  print(('update learning rate: %f -> %f'):format(old_lr, lr))
end

local function MakeIm3(im)
  if im:size(2) == 1 then
    local im3 = torch.repeatTensor(im, 1,3,1,1)
    return im3
  else
    return im
  end
end

function CycleGANModel:GetCurrentVisuals(opt, size)
  local visuals = {}
  table.insert(visuals, {img=MakeIm3(self.real_A), label='real_A'})
  table.insert(visuals, {img=MakeIm3(self.fake_B), label='fake_B'})
  table.insert(visuals, {img=MakeIm3(self.rec_A), label='rec_A'})
  if opt.test == 0 and opt.lambda_identity > 0 then
    table.insert(visuals, {img=MakeIm3(self.identity_A), label='identity_A'})
  end
  table.insert(visuals, {img=MakeIm3(self.real_B), label='real_B'})
  table.insert(visuals, {img=MakeIm3(self.fake_A), label='fake_A'})
  table.insert(visuals, {img=MakeIm3(self.rec_B), label='rec_B'})
  if opt.test == 0 and opt.lambda_identity > 0 then
    table.insert(visuals, {img=MakeIm3(self.identity_B), label='identity_B'})
  end
  return visuals
end
