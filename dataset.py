import torch
import torchvision
from torch.utils.data.dataloader import _DataLoaderIter

class LFADS_MultiSession_Dataset(torch.utils.data.Dataset):
    
    def __init__(self, data_list, device='cpu'):
        super(LFADS_MultiSession_Dataset, self).__init__()
        
        self.data_list   = data_list
        self.device      = device
        self.tensor_list = []
        
        for data in self.data_list:
            self.tensor_list.append(torch.Tensor(data).to(self.device))
            
    def __getitem__(self, ix):
        try:
            return self.tensor_list[ix], ix
        except KeyError:
            raise StopIteration
            
    def __len__(self):
        return len(self.tensor_list)
    
default_collate = torch.utils.data.dataloader._utils.collate.default_collate

class SessionLoader(torch.utils.data.DataLoader):
    
    def __init__(self, dataset, session_size=1, shuffle=False, sampler=None, batch_sampler=None, num_workers=0, collate_fn=default_collate, pin_memory=False, drop_last=False, timeout=0, worker_init_fn=None):
        
        super(SessionLoader, self).__init__(dataset=dataset,
                                            batch_size=session_size,
                                            shuffle=shuffle,
                                            sampler=sampler,
                                            batch_sampler=batch_sampler,
                                            num_workers=num_workers,
                                            collate_fn=collate_fn,
                                            pin_memory=pin_memory,
                                            drop_last=drop_last,
                                            timeout=timeout,
                                            worker_init_fn=worker_init_fn)
        
    def __iter__(self):
        return _SessionLoaderIter(self)
    
class _SessionLoaderIter(_DataLoaderIter):
    
    def __init__(self, loader):
        super(_SessionLoaderIter, self).__init__(loader)
        
    def __next__(self):
        x, idx = super(_SessionLoaderIter, self).__next__()
        x = x.squeeze()
        setattr(x, 'session', idx)
        return x,