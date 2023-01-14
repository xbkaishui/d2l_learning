import torch
from torch import nn
# from d2l import torch as d2l
from loguru import logger


def corr2d(X, K):  # @save
    """计算二维互相关运算"""
    h, w = K.shape
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    logger.info("Y shape {}", Y.shape)
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i:i + h, j:j + w] * K).sum()
    return Y


def test_corr2d():
    X = torch.tensor([[0.0, 1.0, 2.0, 3.0], [3.0, 4.0, 5.0, 6.0], [6.0, 7.0, 8.0, 9.0], [6.0, 7.0, 8.0, 9.0]])
    logger.info("X shape {}", X.shape)
    K = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
    logger.info("kernel shape {}", K.shape)
    corr2d(X, K)
    return None


class Conv2D(nn.Module):
    def __init__(self, kernel_size):
        """
        卷积层对输入和卷积核权重进行互相关运算，并在添加标量偏置之后产生输出。
        所以，卷积层中的两个被训练的参数是卷积核权重和标量偏置。
        就像我们之前随机初始化全连接层一样，在训练基于卷积层的模型时，我们也随机初始化卷积核权重。
        :param kernel_size:
        """
        super().__init__()
        self.weight = nn.Parameter(torch.rand(kernel_size))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return corr2d(x, self.weight) + self.bias


def test_edge_detect():
    # detect edge
    X = torch.ones((6, 8))
    X[:, 2:6] = 0
    logger.info("X {}", X)
    K = torch.tensor([[1.0, -1.0]])
    logger.info("K type {}", K.shape)
    Y = corr2d(X, K)
    logger.info("Y {}", Y)
    # 无法检测水平边缘
    logger.info("X_ver {}", X.t())
    Y_ver = corr2d(X.t(), K)
    logger.info("Y_ver {}", Y_ver)


def test_kernel_grad():
    # 构造一个二维卷积层，它具有1个输出通道和形状为（1，2）的卷积核
    conv2d = nn.Conv2d(1, 1, kernel_size=(1, 2), bias=False)
    X = torch.rand([6, 8])
    Y = torch.rand([6, 7])
    logger.info("x {}", X)
    logger.info("y {}", Y)
    # 这个二维卷积层使用四维输入和输出格式（批量大小、通道、高度、宽度），
    # 其中批量大小和通道数都为1
    X = X.reshape((1, 1, 6, 8))
    Y = Y.reshape((1, 1, 6, 7))
    lr = 3e-2  # 学习率

    for i in range(100):
        Y_hat = conv2d(X)
        l = (Y_hat - Y) ** 2
        conv2d.zero_grad()
        l.sum().backward()
        # 迭代卷积核
        conv2d.weight.data[:] -= lr * conv2d.weight.grad
        if (i + 1) % 2 == 0:
            print(f'epoch {i + 1}, loss {l.sum():.3f}')

def test_conv_1d():
    m = nn.Conv1d(16, 33, 3, stride=3)
    input = torch.randn(20, 16, 50)
    output = m(input)
    logger.info("output shape {}", output.shape)


def test_conv_mul_chl():
    in_channels = 6  # 输入通道数量
    out_channels = 10  # 输出通道数量
    width = 100  # 每个输入通道上的卷积尺寸的宽
    height = 100  # 每个输入通道上的卷积尺寸的高
    kernel_size = 3  # 每个输入通道上的卷积尺寸
    batch_size = 1  # 批数量

    input_tensor = torch.randn(batch_size, in_channels, width, height)
    conv_layer = torch.nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=2)

    out_put = conv_layer(input_tensor)

    logger.info("input shape {}", input_tensor.shape)
    logger.info("output shape {}", out_put.shape)
    logger.info("conv layer shape {}", conv_layer.weight.shape)
    logger.info("bias shape {}", conv_layer.bias.shape)

    logger.info("output {}", out_put)
