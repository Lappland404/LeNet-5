import torch
import torch.nn as nn

class LeNet_5_model(nn.Module):
    def __init__(self):
        super(LeNet_5_model, self).__init__()
    #     self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=0)
    #     self.relu1 = nn.ReLU()
    #     self.pooling1 = nn.AvgPool2d(kernel_size= 2 , stride= 2, padding= 0)
    #     self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=0)
    #     self.relu2 = nn.ReLU()
    #     self.pooling2 = nn.AvgPool2d(kernel_size=2, stride=2)
    #     self.fc1 = nn.Linear(in_features=32 * 6 * 6, out_features=256)
    #     self.relu3 = nn.ReLU()
    #     self.fc2 = nn.Linear(in_features=256, out_features=128)
    #     self.relu4 = nn.Linear(in_features=128, out_features=10)
    #
    # def forward(self,x):
    #     x = self.conv1(x)
    #     x = self.relu1(x)
    #     x = self.pooling1(x)
    #     x = self.conv2(x)
    #     x = self.relu2(x)
    #     x = self.pooling2(x)
    #     x = self.fc1(x)
    #     x = self.relu3(x)
    #     x = self.fc2(x)
    #     out = self.relu4(x)
    #     return out
        ## 定义第一个卷积层
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,  ## 输入的feature map
                out_channels=16,  ## 输出通道数
                kernel_size=3,  ##卷积核尺寸
                stride=1,  ##卷积核步长
                padding=1,  # 进行填充
            ),  ## 卷积后： (1*28*28) ->(16*28*28) （经典311结构 不改变宽高 ）
            nn.ReLU(),  # 激活函数
            nn.AvgPool2d(
                kernel_size=2,  ## 平均值池化层,使用 2*2
                stride=2,  ## 池化步长为2
            ),  ## 池化后：(16*28*28)->(16*14*14)
        )
        ## 定义第二个卷积层
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, 1, 0),  ## 卷积操作(16*14*14)->(32*12*12)  计算过程：（14+2*0-3）/1+1=12
            nn.ReLU(),  # 激活函数
            nn.AvgPool2d(2, 2)  ## 最大值池化操作(32*12*12)->(32*6*6)
        )
        ## 定义全连接层
        self.classifier = nn.Sequential(
            nn.Linear(32 * 6 * 6, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    ## 定义网络的向前传播路径
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)  # 展平多维的卷积图层，经过卷积操作进入全连接层都可能会使用此操作
        output = self.classifier(x)
        return output

