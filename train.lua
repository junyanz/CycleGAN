-- usage example: DATA_ROOT=/path/to/data/ which_direction=BtoA name=expt1 th train.lua
-- code derived from https://github.com/soumith/dcgan.torch and https://github.com/phillipi/pix2pix

require 'torch'
require 'nn'
require 'optim'
util = paths.dofile('util/util.lua')
content = paths.dofile('util/content_loss.lua')
require 'image'
require 'models.architectures'

-- load configuration file
options = require 'options'
opt = options.parse_options('train')

-- setup visualization
visualizer = require 'util/visualizer'

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

-- set batch/instance normalization
set_normalization(opt.norm)

--- timer
local epoch_tm = torch.Timer()
local tm = torch.Timer()

-- define model
local model = nil
local display_plot = nil
if opt.model == 'cycle_gan' then
  assert(data_loader:name() == 'UnalignedDataLoader')
  require 'models.cycle_gan_model'
  model = CycleGANModel()
elseif opt.model == 'pix2pix' then
  require 'models.pix2pix_model'
  assert(data_loader:name() == 'AlignedDataLoader')
  model = Pix2PixModel()
elseif opt.model == 'bigan' then
  assert(data_loader:name() == 'UnalignedDataLoader')
  require 'models.bigan_model'
  model = BiGANModel()
elseif opt.model == 'content_gan' then
  require 'models.content_gan_model'
  assert(data_loader:name() == 'UnalignedDataLoader')
  model = ContentGANModel()
else
  error('Please specify a correct model')
end

-- print the model name
print('Model ' .. model:model_name() .. ' was specified.')
model:Initialize(opt)

-- set up the loss plot
require 'util/plot_util'
plotUtil = PlotUtil()
display_plot = model:DisplayPlot(opt)
plotUtil:Initialize(display_plot, opt.display_id, opt.name)

--------------------------------------------------------------------------------
-- Helper Functions
--------------------------------------------------------------------------------
function visualize_current_results()
  local visuals = model:GetCurrentVisuals(opt)
  for i,visual in ipairs(visuals) do
    visualizer.disp_image(visual.img, opt.display_winsize,
                          opt.display_id+i, opt.name .. ' ' .. visual.label)
  end
end

function save_current_results(epoch, counter)
  local visuals = model:GetCurrentVisuals(opt)
  for i,visual in ipairs(visuals) do
    output_path = paths.concat(opt.visual_dir, 'train_epoch' .. epoch .. '_iter' .. counter .. '_' .. visual.label .. '.jpg')
    visualizer.save_results(visual.img, output_path)
  end
end

function print_current_errors(epoch, counter_in_epoch)
  print(('Epoch: [%d][%8d / %8d]\t Time: %.3f  DataTime: %.3f  '
           .. '%s'):
      format(epoch, ((counter_in_epoch-1) / opt.batchSize),
      math.floor(math.min(data_loader:size(), opt.ntrain) / opt.batchSize),
      tm:time().real / opt.batchSize,
      data_loader:time_elapsed_to_fetch_data() / opt.batchSize,
      model:GetCurrentErrorDescription()
  ))
end

function plot_current_errors(epoch, counter_ratio, opt)
  local errs = model:GetCurrentErrors(opt)
  local plot_vals = { epoch + counter_ratio}
  plotUtil:Display(plot_vals, errs)
end

--------------------------------------------------------------------------------
-- Main Training Loop
--------------------------------------------------------------------------------
local counter = 0
local num_batches = math.floor(math.min(data_loader:size(), opt.ntrain) / opt.batchSize)
print('#training iterations: ' .. opt.niter+opt.niter_decay )

for epoch = 1, opt.niter+opt.niter_decay do
    epoch_tm:reset()
    for counter_in_epoch = 1, math.min(data_loader:size(), opt.ntrain), opt.batchSize do
        tm:reset()
        -- load a batch and run G on that batch
        local real_dataA, real_dataB, _, _ = data_loader:GetNextBatch()

        model:Forward({real_A=real_dataA, real_B=real_dataB}, opt)
        -- run forward pass
        opt.counter = counter
        -- run backward pass
        model:OptimizeParameters(opt)
        -- display on the web server
        if counter % opt.display_freq == 0 and opt.display_id > 0 then
          visualize_current_results()
        end

        -- logging
        if counter % opt.print_freq == 0 then
          print_current_errors(epoch, counter_in_epoch)
          plot_current_errors(epoch, counter_in_epoch/num_batches, opt)
        end

        -- save latest model
        if counter % opt.save_latest_freq == 0 and counter > 0 then
          print(('saving the latest model (epoch %d, iters %d)'):format(epoch, counter))
          model:Save('latest', opt)
        end

        -- save latest results
        if counter % opt.save_display_freq == 0 then
          save_current_results(epoch, counter)
        end
        counter = counter + 1
    end

    -- save model at the end of epoch
    if epoch % opt.save_epoch_freq == 0 then
        print(('saving the model (epoch %d, iters %d)'):format(epoch, counter))
        model:Save('latest', opt)
        model:Save(epoch, opt)
   end
    -- print the timing information after each epoch
    print(('End of epoch %d / %d \t Time Taken: %.3f'):
        format(epoch, opt.niter+opt.niter_decay, epoch_tm:time().real))

    -- update learning rate
    if epoch > opt.niter then
      model:UpdateLearningRate(opt)
    end
    -- refresh parameters
    model:RefreshParameters(opt)
end
