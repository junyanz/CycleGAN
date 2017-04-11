local class = require 'class'
require 'models.base_model'
require 'models.architectures'
require 'util.image_pool'
util = paths.dofile('../util/util.lua')
content = paths.dofile('../util/content_loss.lua')

ContentGANModel = class('ContentGANModel', 'BaseModel')

function ContentGANModel:__init(conf)
  BaseModel.__init(self, conf)
  conf = conf or {}
end

function ContentGANModel:model_name()
  return 'ContentGANModel'
end

function ContentGANModel:InitializeStates()
  local optimState = {learningRate=opt.lr, beta1=opt.beta1,}
  return optimState
end
-- Defines models and networks
function ContentGANModel:Initialize(opt)
  if opt.test == 0 then
    self.fakePool = ImagePool(opt.pool_size)
  end
  -- define tensors
  self.real_A = torch.Tensor(opt.batchSize, opt.input_nc, opt.fineSize, opt.fineSize)
  self.fake_B = torch.Tensor(opt.batchSize, opt.output_nc, opt.fineSize, opt.fineSize)
  self.real_B = self.fake_B:clone() --torch.Tensor(opt.batchSize, opt.output_nc, opt.fineSize, opt.fineSize)

  -- load/define models
  self.criterionGAN = nn.MSECriterion()
  self.criterionContent = nn.AbsCriterion()
  self.contentFunc = content.defineContent(opt.content_loss, opt.layer_name)
  self.netG, self.netD = nil, nil
  if opt.continue_train == 1 then
    if opt.which_epoch then -- which_epoch option exists in test mode
      self.netG = util.load_test_model('G_A', opt)
      self.netD = util.load_test_model('D_A', opt)
    else
      self.netG = util.load_model('G_A', opt)
      self.netD = util.load_model('D_A', opt)
    end
  else
    self.netG = defineG(opt.input_nc, opt.output_nc, opt.ngf, opt.which_model_netG)
    print('netG...', self.netG)
    self.netD = defineD(opt.output_nc, opt.ndf, opt.which_model_netD, opt.n_layers_D, false)
    print('netD...', self.netD)
  end
  -- define real/fake labels
  netD_output_size = self.netD:forward(self.real_A):size()
  self.fake_label = torch.Tensor(netD_output_size):fill(0.0)
  self.real_label = torch.Tensor(netD_output_size):fill(1.0) -- no soft smoothing
  self.optimStateD = self:InitializeStates()
  self.optimStateG = self:InitializeStates()
  self:RefreshParameters()
  print('---------- # Learnable Parameters --------------')
  print(('G = %d'):format(self.parametersG:size(1)))
  print(('D = %d'):format(self.parametersD:size(1)))
  print('------------------------------------------------')
  -- os.exit()
end

-- Runs the forward pass of the network and
-- saves the result to member variables of the class
function ContentGANModel:Forward(input, opt)
  if opt.which_direction == 'BtoA' then
    local temp = input.real_A
    input.real_A = input.real_B
    input.real_B = temp
  end

  self.real_A:copy(input.real_A)
  self.real_B:copy(input.real_B)
  self.fake_B = self.netG:forward(self.real_A):clone()
  -- output = {self.fake_B}
  output =  {}
  -- if opt.test == 1 then

  -- end
  return output
end

-- create closure to evaluate f(X) and df/dX of discriminator
function ContentGANModel:fDx_basic(x, gradParams, netD, netG,
                                   real_target, fake_target, opt)
  util.BiasZero(netD)
  util.BiasZero(netG)
  gradParams:zero()

  local errD_real, errD_rec, errD_fake, errD = 0, 0, 0, 0
  -- Real  log(D_A(B))
  local output = netD:forward(real_target)
  errD_real = self.criterionGAN:forward(output, self.real_label)
  df_do = self.criterionGAN:backward(output, self.real_label)
  netD:backward(real_target, df_do)

  -- Fake  + log(1 - D_A(G_A(A)))
  output = netD:forward(fake_target)
  errD_fake = self.criterionGAN:forward(output, self.fake_label)
  df_do = self.criterionGAN:backward(output, self.fake_label)
  netD:backward(fake_target, df_do)
  errD = (errD_real + errD_fake) / 2.0
  -- print('errD', errD
  return errD, gradParams
end


function ContentGANModel:fDx(x, opt)
  fake_B = self.fakePool:Query(self.fake_B)
  self.errD, gradParams = self:fDx_basic(x, self.gradparametersD, self.netD, self.netG,
                                     self.real_B, fake_B, opt)
  return self.errD, gradParams
end

function ContentGANModel:fGx_basic(x, netG_source, netD_source, real_source, real_target, fake_target,
                                   gradParametersG_source, opt)
  util.BiasZero(netD_source)
  util.BiasZero(netG_source)
  gradParametersG_source:zero()
  -- GAN loss
  -- local df_d_GAN = torch.zeros(fake_target:size())
  -- local errGAN = 0
  -- local errRec = 0
    --- Domain GAN loss: D_A(G_A(A))
  local output = netD_source.output -- [hack] forward was already executed in fDx, so save computation netD_source:forward(fake_B) ---
  local errGAN = self.criterionGAN:forward(output, self.real_label)
  local df_do = self.criterionGAN:backward(output, self.real_label)
  local df_d_GAN = netD_source:updateGradInput(fake_target, df_do) ---:narrow(2,fake_AB:size(2)-output_nc+1, output_nc)

  -- content loss
  -- print('content_loss', opt.content_loss)
  -- function content.lossUpdate(criterionContent, real_source, fake_target, contentFunc, loss_type, weight)
  local errContent, df_d_content = content.lossUpdate(self.criterionContent, real_source, fake_target, self.contentFunc, opt.content_loss, opt.lambda_A)
  netG_source:forward(real_source)
  netG_source:backward(real_source, df_d_GAN  + df_d_content)
  -- print('errD', errGAN)
  return gradParametersG_source, errGAN, errContent
end

function ContentGANModel:fGx(x, opt)
  self.gradparametersG, self.errG, self.errCont =
  self:fGx_basic(x, self.netG, self.netD,
             self.real_A, self.real_B, self.fake_B,
             self.gradparametersG, opt)
  return self.errG, self.gradparametersG
end

function ContentGANModel:OptimizeParameters(opt)
  local fDx = function(x) return self:fDx(x, opt) end
  local fGx = function(x) return self:fGx(x, opt) end
  optim.adam(fDx, self.parametersD, self.optimStateD)
  optim.adam(fGx, self.parametersG, self.optimStateG)
end

function ContentGANModel:RefreshParameters()
  self.parametersD, self.gradparametersD = nil, nil -- nil them to avoid spiking memory
  self.parametersG, self.gradparametersG = nil, nil
  -- define parameters of optimization
  self.parametersG, self.gradparametersG = self.netG:getParameters()
  self.parametersD, self.gradparametersD = self.netD:getParameters()
end

function ContentGANModel:Save(prefix, opt)
  util.save_model(self.netG, prefix .. '_net_G_A.t7', 1.0)
  util.save_model(self.netD, prefix .. '_net_D_A.t7', 1.0)
end

function ContentGANModel:GetCurrentErrorDescription()
  description = ('G: %.4f  D: %.4f  Content: %.4f'):format(self.errG and self.errG or -1,
                         self.errD and self.errD or -1,
                         self.errCont and self.errCont or -1)
  return description
end


function ContentGANModel:GetCurrentErrors()
  local errors = {errG=self.errG and self.errG or -1, errD=self.errD and self.errD or -1,
  errCont=self.errCont and self.errCont or -1}
  return errors
end

-- returns a string that describes the display plot configuration
function ContentGANModel:DisplayPlot(opt)
  return 'errG,errD,errCont'
end


function ContentGANModel:GetCurrentVisuals(opt, size)
  if not size then
    size = opt.display_winsize
  end

  local visuals = {}
  table.insert(visuals, {img=self.real_A, label='real_A'})
  table.insert(visuals, {img=self.fake_B, label='fake_B'})
  table.insert(visuals, {img=self.real_B, label='real_B'})
  return visuals
end
