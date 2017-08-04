
--[[
    This data loader is a modified version of the one from dcgan.torch
    (see https://github.com/soumith/dcgan.torch/blob/master/data/donkey_folder.lua).
    Copyright (c) 2016, Deepak Pathak [See LICENSE file for details]
    Copyright (c) 2015-present, Facebook, Inc.
    All rights reserved.
    This source code is licensed under the BSD-style license found in the
    LICENSE file in the root directory of this source tree. An additional grant
    of patent rights can be found in the PATENTS file in the same directory.
]]--

require 'image'
paths.dofile('dataset.lua')
-- This file contains the data-loading logic and details.
-- It is run by each data-loader thread.
------------------------------------------
-------- COMMON CACHES and PATHS
-- Check for existence of opt.data
if opt.DATA_ROOT then
  opt.data = paths.concat(opt.DATA_ROOT, opt.phase)
else
  print(os.getenv('DATA_ROOT'))
  opt.data = paths.concat(os.getenv('DATA_ROOT'), opt.phase)
end

if not paths.dirp(opt.data) then
    error('Did not find directory: ' .. opt.data)
end

-- a cache file of the training metadata (if doesnt exist, will be created)
local cache_prefix = opt.data:gsub('/', '_')
os.execute(('mkdir -p %s'):format(opt.cache_dir))
local trainCache = paths.concat(opt.cache_dir, cache_prefix .. '_trainCache.t7')

--------------------------------------------------------------------------------------------
local input_nc = opt.nc -- input channels
local loadSize   = {input_nc, opt.loadSize}
local sampleSize = {input_nc, opt.fineSize}

local function loadImage(path)
  local input = image.load(path, 3, 'float')
  local h = input:size(2)
  local w = input:size(3)

  local imA = image.crop(input, 0, 0, w/2, h)
  imA = image.scale(imA, loadSize[2], loadSize[2])
  local imB = image.crop(input, w/2, 0, w, h)
  imB = image.scale(imB, loadSize[2], loadSize[2])

  local perm = torch.LongTensor{3, 2, 1}
  imA = imA:index(1, perm)
  imA = imA:mul(2):add(-1)
  imB = imB:index(1, perm)
  imB = imB:mul(2):add(-1)

  assert(imA:max()<=1,"A: badly scaled inputs")
  assert(imA:min()>=-1,"A: badly scaled inputs")
  assert(imB:max()<=1,"B: badly scaled inputs")
  assert(imB:min()>=-1,"B: badly scaled inputs")


  local oW = sampleSize[2]
  local oH = sampleSize[2]
  local iH = imA:size(2)
  local iW = imA:size(3)

  if iH~=oH then
    h1 = math.ceil(torch.uniform(1e-2, iH-oH))
  end

  if iW~=oW then
    w1 = math.ceil(torch.uniform(1e-2, iW-oW))
  end
  if iH ~= oH or iW ~= oW then
    imA = image.crop(imA, w1, h1, w1 + oW, h1 + oH)
    imB = image.crop(imB, w1, h1, w1 + oW, h1 + oH)
  end

  if opt.flip == 1 and torch.uniform() > 0.5 then
    imA = image.hflip(imA)
    imB = image.hflip(imB)
  end

  local concatenated = torch.cat(imA,imB,1)

  return concatenated
end


local function loadSingleImage(path)
    local im = image.load(path, input_nc, 'float')
    if opt.resize_or_crop == 'resize_and_crop' then
      im = image.scale(im, loadSize[2], loadSize[2])
    end
    if input_nc == 3 then
      local perm = torch.LongTensor{3, 2, 1}
      im = im:index(1, perm)--:mul(256.0): brg, rgb
      im = im:mul(2):add(-1)
    end
    assert(im:max()<=1,"A: badly scaled inputs")
    assert(im:min()>=-1,"A: badly scaled inputs")

    local oW = sampleSize[2]
    local oH = sampleSize[2]
    local iH = im:size(2)
    local iW = im:size(3)
    if (opt.resize_or_crop == 'resize_and_crop' ) then
      local h1, w1 = 0, 0
      if iH~=oH then
        h1 = math.ceil(torch.uniform(1e-2, iH-oH))
      end
      if iW~=oW then
        w1 = math.ceil(torch.uniform(1e-2, iW-oW))
      end
      if iH ~= oH or iW ~= oW then
        im = image.crop(im, w1, h1, w1 + oW, h1 + oH)
      end
    elseif (opt.resize_or_crop == 'combined') then
      local sH = math.min(math.ceil(oH * torch.uniform(1+1e-2, 2.0-1e-2)), iH-1e-2)
      local sW = math.min(math.ceil(oW * torch.uniform(1+1e-2, 2.0-1e-2)), iW-1e-2)
      local h1 = math.ceil(torch.uniform(1e-2, iH-sH))
      local w1 = math.ceil(torch.uniform(1e-2, iW-sW))
      im = image.crop(im, w1, h1, w1 + sW, h1 + sH)
      im = image.scale(im, oW, oH)
    elseif (opt.resize_or_crop == 'crop') then
      local w = math.min(math.min(oH, iH),iW)
      w = math.floor(w/4)*4
      local x = math.floor(torch.uniform(0, iW - w))
      local y = math.floor(torch.uniform(0, iH - w))
      im = image.crop(im, x, y, x+w, y+w)
    elseif (opt.resize_or_crop == 'scale_width') then
      w = oW
      h = torch.floor(iH * oW/iW)
      im = image.scale(im, w, h)
    elseif (opt.resize_or_crop == 'scale_height') then
      h = oH
      w = torch.floor(iW * oH / iH)
      im = image.scale(im, w, h)
    end

    if opt.flip == 1 and torch.uniform() > 0.5 then
        im = image.hflip(im)
    end

  return im

end

-- channel-wise mean and std. Calculate or load them from disk later in the script.
local mean,std
--------------------------------------------------------------------------------
-- Hooks that are used for each image that is loaded

-- function to load the image, jitter it appropriately (random crops etc.)
local trainHook_singleimage = function(self, path)
   collectgarbage()
  --  print('load single image')
   local im = loadSingleImage(path)
   return im
end

-- function that loads images that have juxtaposition
-- of two images from two domains
local trainHook_doubleimage = function(self, path)
  -- print('load double image')
  collectgarbage()

  local im = loadImage(path)
  return im
end


if opt.align_data > 0 then
  sample_nc = input_nc*2
  trainHook = trainHook_doubleimage
else
  sample_nc = input_nc
  trainHook = trainHook_singleimage
end

trainLoader = dataLoader{
    paths = {opt.data},
    loadSize = {input_nc, loadSize[2], loadSize[2]},
    sampleSize = {sample_nc, sampleSize[2], sampleSize[2]},
    split = 100,
    serial_batches = opt.serial_batches,
    verbose = true
 }

trainLoader.sampleHookTrain = trainHook
collectgarbage()

-- do some sanity checks on trainLoader
do
   local class = trainLoader.imageClass
   local nClasses = #trainLoader.classes
   assert(class:max() <= nClasses, "class logic has error")
   assert(class:min() >= 1, "class logic has error")
end
