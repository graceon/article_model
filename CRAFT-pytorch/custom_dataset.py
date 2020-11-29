import torch
from torch import unsqueeze
import random


# class custom_dataset(torch.utils.data.Dataset):
#     def __init__(self,set_path,shuffle=True):
#         self.dataset=torch.load(set_path)
#         self.len=len(self.dataset)
#         self.index_shuffle=[i for i in range(self.len)]
#         if shuffle:
#         	random.shuffle(self.index_shuffle)
#     def __getitem__(self, index):
#         if index>=self.len:
#             raise StopIteration
#         # print(self.dataset[index][1].long())
#         # print(self.index_shuffle[index])
#         item=self.dataset[self.index_shuffle[index]]
#         x,y=(unsqueeze(item[0].float(),0),item[1].long(),item[2].long())
#         # print (x.shape,y.shape)
#         return (x,y)
#     def __len__(self):
#    		return self.len
# class custom_dataset_print(torch.utils.data.Dataset):
#     def __init__(self,set_path,shuffle=True):
#         self.dataset=torch.load(set_path)
#         self.len=len(self.dataset)
#         self.index_shuffle=[i for i in range(self.len)]
#         if shuffle:
#         	random.shuffle(self.index_shuffle)
#     def __getitem__(self, index):
#         if index>=self.len:
#             raise StopIteration
#         # print(self.dataset[index][1].long())
#         # print(self.index_shuffle[index])
#         item=self.dataset[self.index_shuffle[index]]
#         x,y=(unsqueeze(item[0].float(),0),item[1].long())
#         # print (x.shape,y.shape)
#         return (x,y)
#     def __len__(self):
#    		return self.len
class custom_dataset(torch.utils.data.Dataset):
    def __init__(self, set_path, shuffle=True):
        self.dataset_x, self.dataset_y = torch.load(set_path)
        if len(self.dataset_x) != len(self.dataset_y):
            print('error:', len(self.dataset_x), len(self.dataset_y))
        self.len = len(self.dataset_x)
        print('custom set len:', self.len)
        self.index_shuffle = [i for i in range(self.len)]
        if shuffle:
            random.shuffle(self.index_shuffle)

    def __getitem__(self, index):
        if index >= self.len:
            raise StopIteration
        # print(self.dataset[index][1].long())
        # print(self.index_shuffle[index])
        # self.dataset_x[self.index_shuffle[index]]
        # x,y=(unsqueeze(item[0].float(),0),item[1].long(),item[2].long())
        # print (x.shape,y.shape)
        index_sf = self.index_shuffle[index]
        return (self.dataset_x[index_sf].unsqueeze(0).float(), self.dataset_y[index_sf].long())

    def __len__(self):
        return self.len
