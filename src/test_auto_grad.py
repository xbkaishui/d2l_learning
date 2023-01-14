import torch
from loguru import logger


def simple_grade():
    x = torch.arange(4, dtype=torch.float64)
    x.requires_grad_(True)
    logger.info("x grad {}, x {}", x.grad, x)

    # y = 2 * x ^2
    # grad = 4x
    y = 2 * torch.dot(x, x)
    logger.info(y)

    y.backward()
    logger.info("x grad {}", x.grad)

    flag = (x.grad == 4 * x)
    logger.info("flag {}", flag)

    # 在默认情况下，PyTorch会累积梯度，我们需要清除之前的值
    x.grad.zero_()
    y = x.sum()
    y.backward()
    logger.info("after sum y {}", y)
    logger.info("x grad {}", x.grad)


def mult_dim_grad():
    import torch
    import matplotlib.pyplot as plt
    x = torch.arange(0.0, 10.0, 0.1)
    x.requires_grad_(True)
    x1 = x.detach()
    y1 = torch.sin(x1)
    y2 = torch.sin(x)
    y2.sum().backward()
    plt.plot(x1, y1)
    # plt.plot(x1, x.grad)
    plt.show()


mult_dim_grad()