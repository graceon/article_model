import torch
import os
from PIL import Image
import matplotlib.pyplot as plt
import config
import cv2
import numpy as np

from transform_utils import convert_to_binary_inv,convert_to_binary,adapt_size
from torchvision import datasets, transforms

ToTensor=transforms.ToTensor()
# dataset_train_group_end_collection=[161,370,399]
# dataset_train_path='/fdisk/liver/train/'

# dataset_val_group_end_collection=[19]
# dataset_val_path='/fdisk/liver/val/'

# output_train='./output/train/'
# output_val='./output/val/'





#     datasets.MNIST('../data', train=True, download=True,
#                    transform=transforms.Compose([
#                        transforms.ToTensor(),
#                        transforms.Normalize((0.1307,), (0.3081,))
#                    ])),







# def stack_png(path,start,end):
# 	x=list()
# 	y=list()
# 	for i in range(start,end+1):
# 		x.append(ToTensor(Image.open(path+('%03d'%i)+'.png'))[0]*255)
# 		y.append(ToTensor(Image.open(path+('%03d'%i)+'_mask.png'))[0]*255)
# 	x=torch.stack(x,dim=0)
# 	x=torch.tensor(x,dtype=torch.uint8)
# 	y=torch.stack(y,dim=0)
# 	y=torch.tensor(y,dtype=torch.uint8)
# 	return x,y
# if __name__ =="__main__":


# 	last_end=0
# 	group_index=0
# 	for i in dataset_train_group_end_collection:
# 		x,y=stack_png(dataset_train_path,last_end,i)
# 		torch.save(x,output_train+str(group_index)+'.pt')
# 		torch.save(y,output_train+str(group_index)+'_mask.pt')
# 		print(x.shape,y.shape)
# 		last_end=i+1
# 		group_index+=1
# 	last_end=0
# 	group_index=0
# 	for i in dataset_val_group_end_collection:
# 		x,y=stack_png(dataset_val_path,last_end,i)
# 		torch.save(x,output_val+str(group_index)+'.pt')
# 		torch.save(y,output_val+str(group_index)+'_mask.pt')
# 		print(x.shape,y.shape)
# 		last_end=i+1
# 		group_index+=1


        # item_x=cv2.imread(basepath + item, 0)
        # item_x=ToTensor(Image.open(basepath+item))*255
        # print(item_x.shape,item_y)

global DEBUG
DEBUG=0



dilate_kernel=[np.ones((3,3),np.uint8)]
 



def append_dir(package_x,package_y,basepath,item_y,item_y_base):
    global DEBUG

    factor=100

    path = os.listdir(basepath)
    for item in path:
        item_x=Image.open(basepath + item)

        temp=np.array(item_x)
        if temp.shape[0]!=28 or temp.shape[1]!=28:
            item_x=item_x.resize((28,28),Image.BILINEAR)

        item_x=np.array(item_x)
        item_x = item_x.astype(np.uint8)
        item_x = convert_to_binary(item_x)
        item_x = torch.from_numpy(item_x)
        if (item_x.max()<0.1):
            print(basepath + item)
        item_x/=item_x.max()
        item_x*=255
        for i_factor in range(factor):
            package_x.append(torch.tensor(item_x,dtype=torch.uint8))
            package_y.append(torch.tensor(item_y,dtype=torch.uint8))
        # print(package[-1][1])
        # item_x-=item_x.min()
        # if(item_x.max()==0):
        #     print(basepath + item)
        # print(item_y)
        if DEBUG:
            print(package_y[-1])
            plt.matshow(package_x[-1])
            plt.show()
def append_dir_kaggle(package_x,package_y,basepath,item_y,item_y_base):
    global DEBUG


    path = os.listdir(basepath)
    for item in path:
        item_x=Image.open(basepath + item)

        item_x=np.array(item_x)
        item_x=convert_to_binary_inv(np.array(item_x,dtype=np.uint8))

        item_x = cv2.dilate(item_x,dilate_kernel[0])

        item_x=np.array(item_x,dtype=np.uint8)
        contour=[]


        n_n=45
        for hw in range(n_n*n_n):
            if(item_x[hw%n_n][hw//n_n]==255):
                contour.append([hw%n_n,hw//n_n])

        contour=np.array(contour,dtype=np.int8)

        # print(contour[:,0].min(),contour[:,0].max(),contour[:,1].min(),contour[:,1].max())
        # print(basepath + item)
        # print(item_x.shape)
        item_x=item_x[contour[:,0].min():contour[:,0].max()+1,contour[:,1].min():contour[:,1].max()+1]
        item_x=torch.tensor(item_x,dtype=torch.uint8)
        item_x = adapt_size(item_x)[0]
        item_x/=item_x.max()
        item_x*=255
        package_x.append(torch.tensor(item_x,dtype=torch.uint8))
        package_y.append(torch.tensor(item_y,dtype=torch.uint8))
        # print(package[-1][1])
        # item_x-=item_x.min()
        # if(item_x.max()==0):
        #     print(basepath + item)
        
        # print(item_y)
        if DEBUG:
            print(package_y[-1])
            plt.matshow(package_x[-1])
            plt.show()
if __name__ =="__main__":
    
    package_x=list()
    package_y=list()

    base_dir='D:\\dataset_math\\'
    train_loader = torch.utils.data.DataLoader(datasets.MNIST(base_dir, train=True, download=True,transform=transforms.Compose([transforms.ToTensor()])),batch_size=1, shuffle=False)
    test_loader = torch.utils.data.DataLoader(datasets.MNIST(base_dir, train=False, download=True,transform=transforms.Compose([transforms.ToTensor()])),batch_size=1, shuffle=False)
    
    item_y_base=config.CLASS_base.index('writing')
    item_y_base="is not used"
    def append_mnist(loader):
        global DEBUG
        for item_x,item_y in loader:
            # print(torch.tensor(item_y,dtype=torch.uint8)[0])
            # print(torch.tensor((1-item_x)*255,dtype=torch.uint8)[0][0].shape)
            # plt.matshow(item_x[0][0])
            # plt.show()
            item_x=convert_to_binary(np.array(item_x*255,dtype=np.uint8)[0][0])
            # plt.matshow(item_x)
            # plt.show()
            contour=[]

            for hw in range(28*28):
                if(item_x[hw%28][hw//28]==255):
                    contour.append([hw%28,hw//28])
            contour=np.array(contour,dtype=np.int8)
            # print(contour[:,0].min(),contour[:,0].max(),contour[:,1].min(),contour[:,1].max())
            item_x=item_x[contour[:,0].min():contour[:,0].max()+1,contour[:,1].min():contour[:,1].max()+1]
            item_x=torch.tensor(item_x,dtype=torch.uint8)
            item_x = adapt_size(item_x)[0]
            item_x/=item_x.max()
            item_x*=255
            package_x.append(torch.tensor(item_x,dtype=torch.uint8))
            package_y.append(torch.tensor(item_y[0],dtype=torch.uint8))
            if DEBUG:
                print(package_y[-1])
                plt.matshow(package_x[-1])
                plt.show()
    if 1:
        append_mnist(train_loader)
        append_mnist(test_loader)
    print(len(package_x))
    print(len(package_y))
    # item_y_base=config.CLASS_base.index('writing')
    # append_dir(package,base_dir+"train/+/",config.CLASS.index('add'),item_y_base)
    # append_dir(package,base_dir+"train/-/",config.CLASS.index('sub'),item_y_base)
    # append_dir(package,base_dir+"train/times/",config.CLASS.index('mul'),item_y_base)
    # append_dir(package,base_dir+"train/div/",config.CLASS.index('div'),item_y_base)
    # append_dir(package,base_dir+"train/=/",config.CLASS.index('equal'),item_y_base)
    if 1:
        base_dir1=base_dir+'html_gen/'
        append_dir(package_x,package_y,base_dir1+"add/",config.CLASS.index('add'),item_y_base)
        append_dir(package_x,package_y,base_dir1+"sub/",config.CLASS.index('sub'),item_y_base)
        append_dir(package_x,package_y,base_dir1+"mul/",config.CLASS.index('mul'),item_y_base)
        append_dir(package_x,package_y,base_dir1+"div/",config.CLASS.index('div'),item_y_base)
        append_dir(package_x,package_y,base_dir1+"equal/",config.CLASS.index('equal'),item_y_base)
        append_dir(package_x,package_y,base_dir1+"0/",config.CLASS.index('0'),item_y_base)
        append_dir(package_x,package_y,base_dir1+"1/",config.CLASS.index('1'),item_y_base)
        append_dir(package_x,package_y,base_dir1+"2/",config.CLASS.index('2'),item_y_base)
        append_dir(package_x,package_y,base_dir1+"3/",config.CLASS.index('3'),item_y_base)
        append_dir(package_x,package_y,base_dir1+"4/",config.CLASS.index('4'),item_y_base)
        append_dir(package_x,package_y,base_dir1+"5/",config.CLASS.index('5'),item_y_base)
        append_dir(package_x,package_y,base_dir1+"6/",config.CLASS.index('6'),item_y_base)
        append_dir(package_x,package_y,base_dir1+"7/",config.CLASS.index('7'),item_y_base)
        append_dir(package_x,package_y,base_dir1+"8/",config.CLASS.index('8'),item_y_base)
        append_dir(package_x,package_y,base_dir1+"9/",config.CLASS.index('9'),item_y_base)
    if 1:
        base_dir1=base_dir+'hand_collect/'
        append_dir(package_x,package_y,base_dir1+"add/",config.CLASS.index('add'),item_y_base)
        append_dir(package_x,package_y,base_dir1+"sub/",config.CLASS.index('sub'),item_y_base)
        append_dir(package_x,package_y,base_dir1+"mul/",config.CLASS.index('mul'),item_y_base)
        append_dir(package_x,package_y,base_dir1+"div/",config.CLASS.index('div'),item_y_base)
        append_dir(package_x,package_y,base_dir1+"equal/",config.CLASS.index('equal'),item_y_base)
        append_dir(package_x,package_y,base_dir1+"0/",config.CLASS.index('0'),item_y_base)
        append_dir(package_x,package_y,base_dir1+"1/",config.CLASS.index('1'),item_y_base)
        append_dir(package_x,package_y,base_dir1+"2/",config.CLASS.index('2'),item_y_base)
        append_dir(package_x,package_y,base_dir1+"3/",config.CLASS.index('3'),item_y_base)
        append_dir(package_x,package_y,base_dir1+"4/",config.CLASS.index('4'),item_y_base)
        append_dir(package_x,package_y,base_dir1+"5/",config.CLASS.index('5'),item_y_base)
        append_dir(package_x,package_y,base_dir1+"6/",config.CLASS.index('6'),item_y_base)
        append_dir(package_x,package_y,base_dir1+"7/",config.CLASS.index('7'),item_y_base)
        append_dir(package_x,package_y,base_dir1+"8/",config.CLASS.index('8'),item_y_base)
        append_dir(package_x,package_y,base_dir1+"9/",config.CLASS.index('9'),item_y_base)
    if 1:
        for i in range(0,10):
            append_dir_kaggle(package_x,package_y,base_dir+str(i)+"/",config.CLASS.index(str(i)),item_y_base)
    torch.save([package_x,package_y],'package.pth')