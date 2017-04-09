-- usage: DATA_ROOT=/path/to/data/ name=expt1 which_direction=BtoA th test.lua
--
-- code derived from https://github.com/soumith/dcgan.torch and https://github.com/phillipi/pix2pix
require 'image'
require 'nn'
require 'nngraph'
require 'models.architectures'


util = paths.dofile('util/util.lua')
options = require 'options'
opt = options.parse_options('test')

-- initialize torch GPU/CPU mode
if opt.gpu > 0 then
  require 'cutorch'
  require 'cunn'
  cutorch.setDevice(opt.gpu)
  print ("GPU Mode")
  torch.setdefaulttensortype('torch.CudaTensor')
else
  torch.setdefaulttensortype('torch.FloatTensor')
  print ("CPU Mode")
end

-- setup visualization
visualizer = require 'util/visualizer'

function TableConcat(t1,t2)
  for i=1,#t2 do
    t1[#t1+1] = t2[i]
  end
  return t1
end


-- load data
local data_loader = nil
if opt.align_data > 0 then
  require 'data.aligned_data_loader'
  data_loader = AlignedDataLoader()
else
  require 'data.unaligned_data_loader'
  data_loader = UnalignedDataLoader()
end
print( "DataLoader " .. data_loader:name() .. " was created.")
data_loader:Initialize(opt)

if opt.how_many == 'all' then
  opt.how_many = data_loader:size()
end

opt.how_many = math.min(opt.how_many, data_loader:size())

-- set batch/instance normalization
set_normalization(opt.norm)

-- load model
opt.continue_train = 1
-- define model
if opt.model == 'cycle_gan' then
  require 'models.cycle_gan_model'
  model  = CycleGANModel()
elseif opt.model == 'one_direction_test' then
  require 'models.one_direction_test_model'
  model = OneDirectionTestModel()
elseif opt.model == 'pix2pix' then
  require 'models.pix2pix_model'
  model = Pix2PixModel()
elseif opt.model == 'bigan' then
  require 'models.bigan_model'
  model  = BiGANModel()
elseif opt.model == 'content_gan' then
  require 'models.content_gan_model'
  model = ContentGANModel()
else
  error('Please specify a correct model')
end
model:Initialize(opt)

local pathsA = {} -- paths to images A tested on
local pathsB = {} -- paths to images B tested on
local web_dir = paths.concat(opt.results_dir, opt.name .. '/' .. opt.which_epoch .. '_' .. opt.phase)
paths.mkdir(web_dir)
local image_dir = paths.concat(web_dir, 'images')
paths.mkdir(image_dir)
s1 = opt.fineSize
s2 = opt.fineSize / opt.aspect_ratio

visuals = {}

for n = 1, math.floor(opt.how_many) do
  print('processing batch ' .. n)
  local cur_dataA, cur_dataB, cur_pathsA, cur_pathsB = data_loader:GetNextBatch()

  cur_pathsA = util.basename_batch(cur_pathsA)
  cur_pathsB = util.basename_batch(cur_pathsB)
  print('pathsA', cur_pathsA)
  print('pathsB', cur_PathsB)
  model:Forward({real_A=cur_dataA, real_B=cur_dataB}, opt)

  visuals = model:GetCurrentVisuals(opt, opt.fineSize)

  for i,visual in ipairs(visuals) do
    if opt.resize_or_crop == 'scale_width' or opt.resize_or_crop == 'scale_height' then
      s1 = nil
      s2 = nil
    end
    visualizer.save_images(visual.img, paths.concat(image_dir, visual.label), {string.gsub(cur_pathsA[1],'.jpg','.png')}, s1, s2)
  end


  print('Saved images to: ', image_dir)
  pathsA = TableConcat(pathsA, cur_pathsA)
  pathsB = TableConcat(pathsB, cur_pathsB)
end

labels = {}
for i,visual in ipairs(visuals) do
  table.insert(labels, visual.label)
end

-- make webpage
io.output(paths.concat(web_dir, 'index.html'))
io.write('<table style="text-align:center;">')
io.write('<tr><td> Image </td>')
for i = 1, #labels do
  io.write('<td>' .. labels[i] .. '</td>')
end
io.write('</tr>')

for n = 1,math.floor(opt.how_many) do
  io.write('<tr>')
  io.write('<td>' .. tostring(n) .. '</td>')
  for j = 1, #labels do
    label = labels[j]
    io.write('<td><img src="./images/' .. label .. '/' .. string.gsub(pathsA[n],'.jpg','.png') .. '"/></td>')
  end
  io.write('</tr>')
end

io.write('</table>')
