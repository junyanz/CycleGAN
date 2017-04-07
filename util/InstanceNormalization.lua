require 'nn'

--[[
  Implements instance normalization as described in the paper

  Instance Normalization: The Missing Ingredient for Fast Stylization
  Dmitry Ulyanov, Andrea Vedaldi, Victor Lempitsky
  https://arxiv.org/abs/1607.08022
  This implementation is based on
  https://github.com/DmitryUlyanov/texture_nets
]]

local InstanceNormalization, parent = torch.class('nn.InstanceNormalization', 'nn.Module')

function InstanceNormalization:__init(nOutput, eps, momentum, affine)
   parent.__init(self)
   self.running_mean = torch.zeros(nOutput)
   self.running_var = torch.ones(nOutput)

   self.eps = eps or 1e-5
   self.momentum = momentum or 0.0
   if affine ~= nil then
      assert(type(affine) == 'boolean', 'affine has to be true/false')
      self.affine = affine
   else
      self.affine = true
   end

   self.nOutput = nOutput
   self.prev_batch_size = -1

   if self.affine then
      self.weight = torch.Tensor(nOutput):uniform()
      self.bias = torch.Tensor(nOutput):zero()
      self.gradWeight = torch.Tensor(nOutput)
      self.gradBias = torch.Tensor(nOutput)
   end
end

function InstanceNormalization:updateOutput(input)
   self.output = self.output or input.new()
   assert(input:size(2) == self.nOutput)

   local batch_size = input:size(1)

   if batch_size ~= self.prev_batch_size or (self.bn and self:type() ~= self.bn:type())  then
      self.bn = nn.SpatialBatchNormalization(input:size(1)*input:size(2), self.eps, self.momentum, self.affine)
      self.bn:type(self:type())
      self.bn.running_mean:copy(self.running_mean:repeatTensor(batch_size))
      self.bn.running_var:copy(self.running_var:repeatTensor(batch_size))

      self.prev_batch_size = input:size(1)
   end

   -- Get statistics
   self.running_mean:copy(self.bn.running_mean:view(input:size(1),self.nOutput):mean(1))
   self.running_var:copy(self.bn.running_var:view(input:size(1),self.nOutput):mean(1))

   -- Set params for BN
   if self.affine then
      self.bn.weight:copy(self.weight:repeatTensor(batch_size))
      self.bn.bias:copy(self.bias:repeatTensor(batch_size))
   end

   local input_1obj = input:contiguous():view(1,input:size(1)*input:size(2),input:size(3),input:size(4))
   self.output = self.bn:forward(input_1obj):viewAs(input)

   return self.output
end

function InstanceNormalization:updateGradInput(input, gradOutput)
   self.gradInput = self.gradInput or gradOutput.new()

   assert(self.bn)

   local input_1obj = input:contiguous():view(1,input:size(1)*input:size(2),input:size(3),input:size(4))
   local gradOutput_1obj = gradOutput:contiguous():view(1,input:size(1)*input:size(2),input:size(3),input:size(4))

   if self.affine then
      self.bn.gradWeight:zero()
      self.bn.gradBias:zero()
   end

   self.gradInput = self.bn:backward(input_1obj, gradOutput_1obj):viewAs(input)

   if self.affine then
      self.gradWeight:add(self.bn.gradWeight:view(input:size(1),self.nOutput):sum(1))
      self.gradBias:add(self.bn.gradBias:view(input:size(1),self.nOutput):sum(1))
   end
   return self.gradInput
end

function InstanceNormalization:clearState()
   self.output = self.output.new()
   self.gradInput = self.gradInput.new()

   if self.bn then
     self.bn:clearState()
   end
end

function InstanceNormalization:evaluate()
end

function InstanceNormalization:training()
end
