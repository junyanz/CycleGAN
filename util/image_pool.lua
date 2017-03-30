local class = require 'class'
ImagePool= class('ImagePool')

require 'torch'
require 'image'

function ImagePool:__init(pool_size)
  self.pool_size = pool_size
  if pool_size > 0 then
    self.num_imgs = 0
    self.images = {}
  end
end

function ImagePool:model_name()
  return 'ImagePool'
end
-- 
-- function ImagePool:Initialize(pool_size)
--   -- torch.manualSeed(0)
--   -- assert(pool_size > 0)
--   self.pool_size = pool_size
--   if pool_size > 0 then
--     self.num_imgs = 0
--     self.images = {}
--   end
-- end

function ImagePool:Query(image)
  -- print('query image')
  if self.pool_size == 0 then
    -- print('get identical image')
    return image
  end
  if self.num_imgs < self.pool_size then
    -- self.images.insert(image:clone())
    self.num_imgs = self.num_imgs + 1
    self.images[self.num_imgs] = image
    return image
  else
    local p = math.random()
    -- print('p' ,p)
    -- os.exit()
    if p > 0.5 then
      -- print('use old image')
      -- random_id = torch.Tensor(1)
      -- random_id:random(1, self.pool_size)
      local random_id = math.random(self.pool_size)
      -- print('random_id', random_id)
      local tmp = self.images[random_id]:clone()
      self.images[random_id] = image:clone()
      return tmp
    else
      return image
    end

  end

end
