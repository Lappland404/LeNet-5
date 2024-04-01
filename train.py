import torch
from torch import optim, nn
from tqdm import tqdm
from model import LeNet_5_model

from dataloder import LN5_Dataloader

def train(net, device, datapath, num_epoch=5, batch_size = 8,lr = 0.0003):
    LeNet5_data = LN5_Dataloader(data_path)
    #加载训练数据
    train_loader = torch.utils.data.DataLoader(dataset=LeNet5_data, batch_size=batch_size, shuffle=True)
    # 使用Adam优化器
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=1e-8)
    # 定义Loss算法
    criterion = nn.CrossEntropyLoss()
    # best_loss统计，初始化为正无穷
    best_loss = float('inf')
    for epoch in range(num_epoch):
        # 训练模式
        net.train()
        # 按照batch_size开始训练
        loop = tqdm(enumerate(train_loader), total=batch_size - 1)
        for step, (image, label) in loop:
            # 梯度清零的方法
            optimizer.zero_grad()
            # 将数据拷贝到device中
            image = image.to(device=device, dtype=torch.float32)
            label = label.to(device=device, dtype=torch.float32)
            # 使用网络参数，输出预测结果
            pred = net(image)
            # 计算loss
            # 在计算损失之前将模型输出转换为整数类型
            pred_classes = torch.argmax(pred, dim=1)  # 找到每个样本预测的类别索引
            loss = criterion(pred, label.long())  # 将目标转换为整数类型
            print('Loss/train', loss.item())
            # 保存loss值最小的网络参数
            if loss < best_loss:
                best_loss = loss
                torch.save(net.state_dict(), 'best_model.pth')
            # 更新参数
            loss.backward()
            optimizer.step()
            loop.set_description(f'Epoch [{epoch}/{num_epoch - 1}]')


if __name__ == "__main__":
    # 选择设备，有cuda用cuda，没有就用cpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 加载网络，图片单通道1，分类为1。
    net = LeNet_5_model()
    # 将网络拷贝到deivce中
    net.to(device=device)
    # 指定训练集地址，开始训练
    data_path = "./datasets"
    train(net, device, data_path)





