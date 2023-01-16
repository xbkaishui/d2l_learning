import torch
from loguru import logger
from torch import nn
from d2l import torch as d2l

net = nn.Sequential(
    # input [1, 1, 224, 224]
    # 这里使用一个11*11的更大窗口来捕捉对象。
    # 同时，步幅为4，以减少输出的高度和宽度。
    # 另外，输出通道的数目远大于LeNet
    nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=1), nn.ReLU(),
    # output [1, 96, 54, 54]
    nn.MaxPool2d(kernel_size=3, stride=2),
    # output [1, 96, 26, 26]
    # 减小卷积窗口，使用填充为2来使得输入与输出的高和宽一致，且增大输出通道数
    nn.Conv2d(96, 256, kernel_size=5, padding=2), nn.ReLU(),
    # output [1, 96, 26, 26]
    nn.MaxPool2d(kernel_size=3, stride=2),
    # [1, 256, 12, 12]
    # 使用三个连续的卷积层和较小的卷积窗口。
    # 除了最后的卷积层，输出通道的数量进一步增加。
    # 在前两个卷积层之后，汇聚层不用于减少输入的高度和宽度
    nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(),
    nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(),
    nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(),
    # [1, 256, 12, 12]
    nn.MaxPool2d(kernel_size=3, stride=2),
    # [1, 256, 5, 5]
    nn.Flatten(),
    # [1, 6400]
    # 这里，全连接层的输出数量是LeNet中的好几倍。使用dropout层来减轻过拟合
    nn.Linear(6400, 4096), nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(4096, 4096), nn.ReLU(),
    nn.Dropout(p=0.5),
    # 最后是输出层。由于这里使用Fashion-MNIST，所以用类别数为10，而非论文中的1000
    nn.Linear(4096, 10))


def test_alex_net():
    x = torch.randn(1, 1, 224, 224)
    logger.info("add net layer")
    logger.info("input shape {}", x.shape)
    for layer in net:
        x = layer(x)
        logger.info("layer {}, output shape {}", layer.__class__.__name__, x.shape)

    # read image file
    batch_size = 128
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)

    lr, num_epochs = 0.01, 1
    d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())


if __name__ == '__main__':
    test_alex_net()