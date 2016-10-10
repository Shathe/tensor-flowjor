from kaffe.tensorflow import Network

class CIFAR10_full_deploy(Network):
    def setup(self):
        (self.feed('data')
             .conv(5, 5, 32, 1, 1, relu=False, name='conv1')
             .max_pool(3, 3, 2, 2, name='pool1')
             .relu(name='relu1')
             .lrn(1, 1.66666666667e-05, 0.75, name='norm1')
             .conv(5, 5, 32, 1, 1, name='conv2')
             .avg_pool(3, 3, 2, 2, name='pool2')
             .lrn(1, 1.66666666667e-05, 0.75, name='norm2')
             .conv(5, 5, 64, 1, 1, name='conv3')
             .avg_pool(3, 3, 2, 2, name='pool3')
             .fc(10, relu=False, name='ip1')
             .softmax(name='prob'))