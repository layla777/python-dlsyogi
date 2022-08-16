import argparse

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import Chain
from chainer import serializers


class MLP(Chain):
    def __init__(self, n_units):
        super(MLP, self).__init__()
        with self.init_scope():
            self.l1 = L.Linear(None, n_units)
            self.l2 = L.Linear(None, n_units)
            self.l3 = L.Linear(None, 10)

    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        return self.l3(h2)


parser = argparse.ArgumentParser(description='example: MNIST')
parser.add_argument('--batchsize', '-b', type=int, default=100, help='Number of images in each mini-batch')
parser.add_argument('--epoch', '-e', type=int, default=20, help='Number of sweeps over the dataset to train')
parser.add_argument('--unit', '-u', type=int, default=1000, help='Number of units')
parser.add_argument('--gpu', '-g', type=int, default=-1, help='GPU ID (negative value indicates CPU)')
parser.add_argument('--initmodel', '-m', default='', help='Initialize model from given file')
parser.add_argument('--resume', '-r', default='', help='Resume the opimizatiion from snapshot')
args = parser.parse_args()

print(f'GPU: {args.gpu}')
print(f'# unit: {args.unit}')
print(f'# Minibatch-size: {args.batchsize}')
print(f'# epoch: {args.epoch}')

model = MLP(args.unit)

if args.gpu > 0:
    pass

optimizer = chainer.optimizers.SGD()
optimizer.setup(model)

if args.initmodel:
    print('Load model from', args.initmodel)
    serializers.load_npz(args.initmodel, model)

if args.resume:
    print('Load optimizer state from', args.resume)
    serializers.load_npz(args.resume, optimizer)

train, test = chainer.datasets.get_mnist()

train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
test_iter = chainer.iterators.SerialIterator(test, args.batchsize, shuffle=False)

for epoch in range(1, args.epoch + 1):
    sum_loss = 0
    itr = 0

    for i in range(0, len(train), args.batchsize):
        train_batch = train_iter.next()
        x, t = chainer.dataset.concat_examples(train_batch, args.gpu)
        y = model(x)
        model.cleargrads()
        loss = F.softmax_cross_entropy(y, t)
        loss.backward()
        optimizer.update()
        sum_loss += loss.data
        itr += 1

    sum_test_loss = 0
    sum_test_accuracy = 0
    test_itr = 0
    for i in range(0, len(test), args.batchsize):
        test_batch = test_iter.next()
        x_test, t_test = chainer.dataset.concat_examples(test_batch, args.gpu)

        y_test = model(x_test)
        sum_test_loss += F.softmax_cross_entropy(y_test, t_test).data
        sum_test_accuracy += F.accuracy(y_test, t_test).data
        test_itr += 1

    print(
        f'epoch={optimizer.epoch + 1}, train loss={sum_loss / itr}, accuracy={sum_test_loss / test_itr, sum_test_accuracy / test_itr}')

    optimizer.new_epoch()

print('save the model')
serializers.save_npz('mlp.model', model)

print('save the optimizer')
serializers.save_npz('mlp.state', optimizer)
