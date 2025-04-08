# 用上之前的训练框架代码
def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):
    for epoch in range(num_epochs):
        metric = d2l.Accumulator(3)  # 训练损失总和，训练准确度总和，样本数
        for X, y in train_iter:
            # 前向传播
            y_hat = net(X)
            l = loss(y_hat, y)

            if isinstance(updater, torch.optim.Optimizer):
                # 使用 PyTorch 的优化器
                updater.zero_grad()
                l.mean().backward()
                updater.step()
            else:
                # 自定义 updater（如用于从零开始的实现）
                l.sum().backward()
                updater(X.shape[0])

            # 累加损失和准确率
            metric.add(float(l.sum()), d2l.accuracy(y_hat, y), y.numel())
        
        # 每个 epoch 打印测试集准确率
        test_acc = evaluate_accuracy(net, test_iter)
        print(f"epoch {epoch + 1}, loss {metric[0] / metric[2]:.4f}, "
              f"train acc {metric[1] / metric[2]:.3f}, test acc {test_acc:.3f}")
        
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
