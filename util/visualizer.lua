-------------------------------------------------------------
-- Various utilities for visualization through the web server
-------------------------------------------------------------

local visualizer = {}

require 'torch'
disp = nil
print(opt)
if opt.display_id > 0 then -- [hack]: assume that opt already existed
  disp = require 'display'
end
util = require 'util/util'
require 'image'

-- function visualizer
function visualizer.disp_image(img_data, win_size, display_id, title)
  images = util.deprocess_batch(util.scaleBatch(img_data:float(),win_size,win_size))
  disp.image(images, {win=display_id, title=title})
end

function visualizer.save_results(img_data, output_path)
  local tensortype = torch.getdefaulttensortype()
  torch.setdefaulttensortype('torch.FloatTensor')
  local image_out = nil
  local win_size = opt.display_winsize
  images = torch.squeeze(util.deprocess_batch(util.scaleBatch(img_data:float(), win_size, win_size)))

  if images:dim() == 3 then
    image_out = images
  else
    for i = 1,images:size(1) do
      img = images[i]
      if image_out == nil then
        image_out = img
      else
        image_out = torch.cat(image_out, img)
      end
    end
  end
  image.save(output_path, image_out)
  torch.setdefaulttensortype(tensortype)
end

function visualizer.save_images(imgs, save_dir, impaths, s1, s2)
  local tensortype = torch.getdefaulttensortype()
  torch.setdefaulttensortype('torch.FloatTensor')
  batchSize = imgs:size(1)
  imgs_f = util.deprocess_batch(imgs):float()
  paths.mkdir(save_dir)
  for i = 1, batchSize do -- imgs_f[i]:size(2), imgs_f[i]:size(3)/opt.aspect_ratio
    if s1 ~= nil and s2 ~= nil then
      im_s = image.scale(imgs_f[i], s1, s2):float()
    else
      im_s = imgs_f[i]:float()
    end
    img_to_save = torch.FloatTensor(im_s:size()):copy(im_s)
    image.save(paths.concat(save_dir, impaths[i]), img_to_save)
  end
  torch.setdefaulttensortype(tensortype)
end

return visualizer
