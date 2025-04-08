# 用上之前的训练框架代码
import torch
import matplotlib.pyplot as plt

def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):
    """带绘图功能的训练函数"""
    # 三个列表：记录每个epoch的训练损失、训练准确率、测试准确率
    train_losses, train_accuracies, test_accuracies = [], [], []

    for epoch in range(num_epochs):
        net.train()
        metric = d2l.Accumulator(3)  # 训练损失总和，正确预测数，总样本数
        for X, y in train_iter:
            y_hat = net(X)
            l = loss(y_hat, y)

            if isinstance(updater, torch.optim.Optimizer):
                updater.zero_grad()
                l.mean().backward()
                updater.step()
            else:
                l.sum().backward()
                updater(X.shape[0])

            metric.add(float(l.sum()), d2l.accuracy(y_hat, y), y.numel())

        # 每个epoch后的统计
        train_loss = metric[0] / metric[2]
        train_acc = metric[1] / metric[2]
        test_acc = evaluate_accuracy(net, test_iter)

        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)

        print(f"epoch {epoch + 1}, loss {train_loss:.4f}, "
              f"train acc {train_acc:.3f}, test acc {test_acc:.3f}")

    # 🔽 训练结束后绘图！
    epochs = range(1, num_epochs + 1)
    plt.plot(epochs, train_losses, label='train loss', color='blue')
    plt.plot(epochs, train_accuracies, label='train acc', linestyle='--', color='magenta')
    plt.plot(epochs, test_accuracies, label='test acc', linestyle='-.', color='green')
    plt.xlabel('epoch')
    plt.legend()
    plt.grid(True)
    plt.show()

        
def evaluate_accuracy(net, data_iter):
    """评估模型在给定数据集上的准确率"""
    if isinstance(net, torch.nn.Module):
        net.eval()  # 进入评估模式（关闭 dropout、batchnorm 等）
    metric = [0.0, 0.0]  # 正确预测数，总样本数
    with torch.no_grad():
        for X, y in data_iter:
            y_hat = net(X)
            if y_hat.ndim > 1 and y_hat.shape[1] > 1:
                y_hat = y_hat.argmax(axis=1)
            cmp = y_hat.type(y.dtype) == y
            metric[0] += float(cmp.sum())
            metric[1] += y.numel()
    return metric[0] / metric[1]
