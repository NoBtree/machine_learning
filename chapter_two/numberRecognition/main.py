# 实现一个手写数字识别程序
import mnist_reader
import network


# 主函数
if __name__ == '__main__':
    # 读取原始数据处理输出 训练集 验证集 测试集
    train_data, validation_data, test_data = mnist_reader.raw_data_process()
    # 初始化神经网络 (784, 15, 10)
    net = network.Network((784, 15, 10))
    # 训练神经网络
    epochs = 10  # 训练次数
    mini_batch_size = 10  # batch大小
    learning_rate = 0.5  # 学习率
    net.SGD(train_data, epochs, mini_batch_size, learning_rate, test_data=test_data)
    # 测试神经网络
    print("Test times {0}: {1}/{2}(正确识别个数/训练总数)".format(0, net.evaluate(test_data), 10000))


