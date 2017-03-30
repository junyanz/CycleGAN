-- define nn module for VGG postprocessing
local VGG_postprocess, parent = torch.class('nn.VGG_postprocess', 'nn.Module')

function VGG_postprocess:__init()
	parent.__init(self)
end

function VGG_postprocess:updateOutput(input)
  self.output = input:add(1):mul(127.5)
	-- print(self.output:max(), self.output:min())
	if self.output:max() > 255 or self.output:min() < 0 then
		print(self.output:min(), self.output:max())
	end
	-- assert(self.output:min()>=0,"badly scaled inputs")
  -- assert(self.output:max()<=255,"badly scaled inputs")

	local mean_pixel = torch.FloatTensor({103.939, 116.779, 123.68})
	mean_pixel = mean_pixel:reshape(1,3,1,1)
	mean_pixel = mean_pixel:repeatTensor(input:size(1), 1, input:size(3), input:size(4)):cuda()
	self.output:add(-1, mean_pixel)
	return self.output
end

function VGG_postprocess:updateGradInput(input, gradOutput)
	self.gradInput = gradOutput:div(127.5)
	return self.gradInput
end
