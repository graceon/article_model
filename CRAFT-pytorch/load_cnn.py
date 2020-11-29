from densenet import DenseNet
import config
import dir_utils
from custom_dataset import custom_dataset
from train_cnn import test_epoch
import torch
import matplotlib.pyplot as plt




class Classifer_box():
    def __init__(self):
        self.model = get_model()
        device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        self.model = self.model.to(device)
    def eval(self,input):
        res=self.model(input)
        _,pred=res.data.cpu().topk(1, dim=1)
        return pred.squeeze().cpu()


def get_model( depth=100, growth_rate=12, efficient=True, valid_size=5000,
         n_epochs=300, batch_size=64, seed=None):
    """
    A demo to show off training of efficient DenseNets.
    Trains and evaluates a DenseNet-BC on CIFAR-10.

    Args:
        data (str) - path to directory where data should be loaded from/downloaded
            (default $DATA_DIR)
        save (str) - path to save the model to (default /tmp)

        depth (int) - depth of the network (number of convolution layers) (default 40)
        growth_rate (int) - number of features added per DenseNet layer (default 12)
        efficient (bool) - use the memory efficient implementation? (default True)

        valid_size (int) - size of validation set
        n_epochs (int) - number of epochs for training (default 300)
        batch_size (int) - size of minibatch (default 256)
        seed (int) - manually set the random seed (default None)
    """

    depth=config.NET_DEPTH

    if (depth - 4) % 3:
        raise Exception('Invalid depth')
    block_config = [(depth - 4) // 6 for _ in range(3)]

    model = DenseNet(
        growth_rate=growth_rate,
        block_config=block_config,
        num_init_features=growth_rate*2,
        num_classes=config.N_CLASS,
        small_inputs=True,
        efficient=efficient,
    )
    epoch_start=dir_utils.index_max("cnn_model")

    model_path="cnn_model/"+str(epoch_start)+"model.pth"

    print('load cnn model:',model_path)
    if(epoch_start>0):
        model.load_state_dict(torch.load(model_path,map_location=torch.device('cpu')))
    # model.cuda()
    model.eval()
    return model.eval()
    # print(model)
    
    # # Print number of parameters
    # num_params = sum(p.numel() for p in model.parameters())
    # print("Total parameters: ", num_params)


    # Train the model
    # train_set=custom_dataset("package.pth",shuffle=False)
    # train_loader=torch.utils.data.DataLoader(train_set,batch_size=1)
    # model.eval()
    # sample=train_set[1378]
    # x=sample[0].cuda()

    # res=model(torch.unsqueeze(x,0))

    # plt.matshow(torch.tensor(sample[0],dtype=torch.uint8)[0])
    # plt.show()
    # print(res)
    # _,pred=res.data.cpu().topk(1, dim=1)
    # print(_)
    # print(pred)
    # print(sample[1])

    # for batch_idx, (input, target) in enumerate(train_loader):
    #         # Create vaiables
    #         if torch.cuda.is_available():
    #             input = input.cuda()
    #             target = target.cuda()

    #         # compute output
    #         output = model(input)
    #         loss = torch.nn.functional.cross_entropy(output, target)

    #         # measure accuracy and record loss
    #         batch_size = target.size(0)
    #         _, pred = output.data.cpu().topk(1, dim=1)
    #         print(pred,target)



    
    # test_results = test_epoch(
    #     model=model,
    #     loader=train_loader,
    #     is_test=True
    # )
