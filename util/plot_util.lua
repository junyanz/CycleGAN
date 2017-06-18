local class = require 'class'
PlotUtil = class('PlotUtil')


require 'torch'
disp = require 'display'
util = require 'util/util'
require 'image'

local unpack = unpack or table.unpack

function PlotUtil:__init(conf)
  conf = conf or {}
end

function PlotUtil:model_name()
  return 'PlotUtil'
end

function PlotUtil:Initialize(display_plot, display_id, name)
  self.display_plot = string.split(string.gsub(display_plot, "%s+", ""), ",")

  self.plot_config = {
    title = name .. ' loss over time',
    labels = {'epoch', unpack(self.display_plot)},
    ylabel = 'loss',
    win  = display_id,
  }

  self.plot_data = {}
  print('display_opt', self.display_plot)
end


function PlotUtil:Display(plot_vals, loss)
  for k, v in ipairs(self.display_plot) do
    if loss[v] ~= nil then
      plot_vals[#plot_vals + 1] = loss[v]
    end
  end

  table.insert(self.plot_data, plot_vals)
  disp.plot(self.plot_data, self.plot_config)
end
