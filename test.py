import opts
from dataset import VISTDataset
from torch.utils.data import DataLoader

import numpy as np
    
opt = opts.parse_opt()

dataset = VISTDataset(opt)

train_loader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=opt.shuffle, num_workers=opt.workers)


#for iter, batch in enumerate(train_loader):
#    print(batch)
#    break


arr = np.arange(5)

print(arr)

new_arr = arr.view(-1, )

print(new_arr)
