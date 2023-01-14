import torch
from loguru import logger

from torch import nn
from d2l import torch as d2l

net = nn.Sequential(
    # block 1
    nn.Conv2d(1, 6, kernel_size=5, padding=2),
    nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    # block 2
    nn.Conv2d(6, 16, kernel_size=5),
    nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    # flatten
    nn.Flatten(),
    # lr 1
    nn.Linear(16 * 5 * 5, 120),
    nn.Sigmoid(),
    # lr 2
    nn.Linear(120, 84),
    nn.Sigmoid(),
    # output
    nn.Linear(84, 10))

# layer Conv2d, input shape torch.Size([1, 1, 28, 28]), output shape torch.Size([1, 6, 28, 28])
# layer Sigmoid, input shape torch.Size([1, 6, 28, 28]), output shape torch.Size([1, 6, 28, 28])
# layer AvgPool2d, input shape torch.Size([1, 6, 28, 28]), output shape torch.Size([1, 6, 14, 14])
# layer Conv2d, input shape torch.Size([1, 6, 14, 14]), output shape torch.Size([1, 16, 10, 10])
# layer Sigmoid, input shape torch.Size([1, 16, 10, 10]), output shape torch.Size([1, 16, 10, 10])
# layer AvgPool2d, input shape torch.Size([1, 16, 10, 10]), output shape torch.Size([1, 16, 5, 5])
# layer Flatten, input shape torch.Size([1, 16, 5, 5]), output shape torch.Size([1, 400])
# layer Linear, input shape torch.Size([1, 400]), output shape torch.Size([1, 120])
# layer Sigmoid, input shape torch.Size([1, 120]), output shape torch.Size([1, 120])
# layer Linear, input shape torch.Size([1, 120]), output shape torch.Size([1, 84])
# layer Sigmoid, input shape torch.Size([1, 84]), output shape torch.Size([1, 84])
# layer Linear, input shape torch.Size([1, 84]), output shape torch.Size([1, 10])

X = torch.rand(size=(1, 1, 28, 28), dtype=torch.float32)
for layer in net:
    input_shape = X.shape
    X = layer(X)
    logger.info("layer {}, input shape {}, output shape {}", layer.__class__.__name__, input_shape, X.shape)

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)


def evaluate_accuracy_gpu(net, data_iter, device=None):  # @save
    """使用GPU计算模型在数据集上的精度"""
    if isinstance(net, nn.Module):
        net.eval()  # 设置为评估模式
        if not device:
            device = next(iter(net.parameters())).device
    logger.info("device {}", device)
    # 正确预测的数量，总预测的数量
    metric = d2l.Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(X, list):
                # BERT微调所需的（之后将介绍）
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            metric.add(d2l.accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]


def train_ch6(net, train_iter, test_iter, num_epochs, lr, device):
    """用GPU训练模型(在第六章定义)"""

    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)

    net.apply(init_weights)
    logger.info('training on {}', device)
    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                            legend=['train loss', 'train acc', 'test acc'])
    timer, num_batches = d2l.Timer(), len(train_iter)
    for epoch in range(num_epochs):
        logger.info("run epoch {}", epoch)
        # 训练损失之和，训练准确率之和，样本数
        metric = d2l.Accumulator(3)
        net.train()
        for i, (X, y) in enumerate(train_iter):
            timer.start()
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            with torch.no_grad():
                metric.add(l * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])
            timer.stop()
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (train_l, train_acc, None))
        test_acc = evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))
        logger.info("epoch {} loss {}", epoch, train_l)
    print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, '
          f'test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
          f'on {str(device)}')


lr, num_epochs = 0.9, 10
train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())

import torch.utils.model_zoo as model_zoo
import torch.onnx

net.eval()
x = torch.randn(1, 1, 28, 28, requires_grad=True)
torch_out = net(x)
logger.info("inference out {}", torch_out)

torch.onnx.export(net,  # model being run
                  x,  # model input (or a tuple for multiple inputs)
                  "lenet.onnx",  # where to save the model (can be a file or file-like object)
                  export_params=True,  # store the trained parameter weights inside the model file
                  opset_version=10,  # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names=['input'],  # the model's input names
                  output_names=['output'],  # the model's output names
                  dynamic_axes={'input': {0: 'batch_size'},  # variable length axes
                                'output': {0: 'batch_size'}})
