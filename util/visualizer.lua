-------------------------------------------------------------
-- Various utilities for visualization through the web server
-------------------------------------------------------------

local visualizer = {}

require 'torch'
disp = nil

if opt.display_id > 0 then -- [hack]: assume that opt already existed
  disp = require 'display'
end
util = require 'util/util'
require 'image'

-- function visualizer
function visualizer.disp_image(img_data, win_size, display_id, title)
  local tensortype = torch.getdefaulttensortype()
  disp.image(util.deprocess_batch(util.scaleBatch(img_data:float(),win_size,win_size)), {win=display_id, title=title})
  torch.setdefaulttensortype(tensortype)
end

function visualizer.disp_images(imgs, opt)
  local tensortype = torch.getdefaulttensortype()
  disp_imgs = {}
  for i,img in ipairs(imgs) do
    disp_img = util.deprocess_batch(util.scaleBatch(img:float(), opt.win_size, opt.win_size))
    table.insert(disp_imgs, disp_img[1])
  end
  disp.images(disp_imgs, {opt.win_size*3, labels=opt.labels, win=opt.display_id})
  torch.setdefaulttensortype(tensortype)
end
--
function visualizer.save_results(visuals, opt, epoch, counter)
  local tensortype = torch.getdefaulttensortype()
  torch.setdefaulttensortype('torch.FloatTensor')
  local image_out = nil
  local win_size = opt.display_winsize
  for i,visual in ipairs(visuals) do
    im = torch.squeeze(util.deprocess_batch(util.scaleBatch(visual.img:float(), win_size, win_size)))

    if image_out == nil then
      image_out = im
    else
      image_out = torch.cat(image_out, im)
    end
  end

  out_path = paths.concat(opt.checkpoints_dir,  opt.name, 'epoch' .. epoch .. '_iter' .. counter .. '_train_res.png')
  image.save(out_path, image_out)
  torch.setdefaulttensortype(tensortype)
end

function visualizer.save_images(imgs, save_dir, impaths, s1, s2)
  local tensortype = torch.getdefaulttensortype()
  torch.setdefaulttensortype('torch.FloatTensor')
  print('saving images', save_dir)
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
