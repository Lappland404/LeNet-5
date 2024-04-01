import torch
from torchvision import transforms
import torch.utils.data as Data
from torchvision.datasets import FashionMNIST


class LN5_Dataloader(Data.Dataset):
    def __init__(self,datapath):
        self.datapath = datapath
        self.train_data = FashionMNIST(
            root="./datasets",
            train=True,
            transform=transforms.ToTensor(),
            download=False # 如果没下载数据，就下载数据；如果已经下载好，就换为False
        )
        # # 载入数据加载器
        # self.train_loader = Data.DataLoader(
        #     dataset=train_data,
        #     batch_size=64,
        #     shuffle=False,
        #     num_workers=0,  # 设置进程数为0 不然后面可能会报错（根据电脑不同而不同）
        # )
        self.test_data = FashionMNIST(
            root="./datasets",
            train=False,
            transform=transforms.ToTensor(),
            download=False  # 如果没下载数据，就下载数据；如果已经下载好，就换为False
        )
        test_data_x = self.test_data.data.type(torch.FloatTensor) / 255
        test_data_x = torch.unsqueeze(test_data_x, dim=1)  # 之所以要增加一个维度是因为，本来是28X28，缺少通道数，我们一般都是要用通道数的
        test_data_y = self.test_data.targets

    def __len__(self):
        return len(self.train_data)
    def __getitem__(self, index):
        image, label = self.train_data[index]
        return image, label

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 调用cpu或者cuda





