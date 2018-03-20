-- modified from https://github.com/NVIDIA/torch-cudnn/blob/master/convert.lua
-- removed error on nngraph

-- modules that can be converted to nn seamlessly
local layer_list = {
  'BatchNormalization',
  'SpatialBatchNormalization',
  'SpatialConvolution',
  'SpatialCrossMapLRN',
  'SpatialFullConvolution',
  'SpatialMaxPooling',
  'SpatialAveragePooling',
  'ReLU',
  'Tanh',
  'Sigmoid',
  'SoftMax',
  'LogSoftMax',
  'VolumetricBatchNormalization',
  'VolumetricConvolution',
  'VolumetricFullConvolution',
  'VolumetricMaxPooling',
  'VolumetricAveragePooling',
}

-- goes over a given net and converts all layers to dst backend
-- for example: net = cudnn_convert_custom(net, cudnn)
-- same as cudnn.convert with gModule check commented out
function cudnn_convert_custom(net, dst, exclusion_fn)
  return net:replace(function(x)
    --if torch.type(x) == 'nn.gModule' then
    --  io.stderr:write('Warning: cudnn.convert does not work with nngraph yet. Ignoring nn.gModule')
    --  return x
    --end
    local y = 0
    local src = dst == nn and cudnn or nn
    local src_prefix = src == nn and 'nn.' or 'cudnn.'
    local dst_prefix = dst == nn and 'nn.' or 'cudnn.'

    local function convert(v)
      local y = {}
      torch.setmetatable(y, dst_prefix..v)
      if v == 'ReLU' then y = dst.ReLU() end -- because parameters
      for k,u in pairs(x) do y[k] = u end
      if src == cudnn and x.clearDesc then x.clearDesc(y) end
      if src == cudnn and v == 'SpatialAveragePooling' then
        y.divide = true
        y.count_include_pad = v.mode == 'CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING'
      end
      if src == nn and string.find(v, 'Convolution') then
         y.groups = 1
      end
      return y
    end

    if exclusion_fn and exclusion_fn(x) then
      return x
    end
    local t = torch.typename(x)
    if t == 'nn.SpatialConvolutionMM' then
      y = convert('SpatialConvolution')
    elseif t == 'inn.SpatialCrossResponseNormalization' then
      y = convert('SpatialCrossMapLRN')
    else
      for i,v in ipairs(layer_list) do
        if torch.typename(x) == src_prefix..v then
          y = convert(v)
        end
      end
    end
    return y == 0 and x or y
  end)
end
