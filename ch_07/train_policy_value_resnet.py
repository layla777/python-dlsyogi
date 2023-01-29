import argparse
import logging
import os
import pickle
import random
import re

import chainer.functions as F
from chainer import cuda, Variable
from chainer import optimizers, serializers

from pydlshogi.network.policy_value_resnet import PolicyValueResnet
from pydlshogi.read_kifu import *

parser = argparse.ArgumentParser()
parser.add_argument('gamelist_train', type=str, help='train game list')
parser.add_argument('gamelist_test', type=str, help='test game list')
parser.add_argument('--blocks', type=int, default=5, help='Number of resnet blocks')
parser.add_argument('--batchsize', '-b', type=int, default=32, help='Number of positions in each mini-batch')
parser.add_argument('--test_batchsize', type=int, default=512, help='Number of positions in each test mini-batch')
parser.add_argument('--epoch', '-e', type=int, default=1, help='Number of epoch times')
parser.add_argument('--model', type=str, default='model/model_policy_value_resnet', help='Model file name')
parser.add_argument('--state', type=str, default='model/state_policy_value_resnet', help='State file name')
parser.add_argument('--initmodel', '-m', default='', help='Initialize the model from given file')
parser.add_argument('--resume', '-r', default='', help='Resume the optimization from snapshot')
parser.add_argument('--log', default=None, help='log file path')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--eval_interval', '-i', type=int, default=1000, help='eval interval')
args = parser.parse_args()

logging.basicConfig(format='%(asctime)s\t%(levelname)s\t%(message)s', datefmt='%Y-%m-%d %H:%M:%S', filename=args.log,
                    level=logging.DEBUG)

model = PolicyValueResnet(args.blocks)
model.to_gpu()

optimizer = optimizers.MomentumSGD(lr=args.lr)
optimizer.setup(model)

# Init / Resume
if args.initmodel:
    logging.info(f'Loading model from {args.initmodel}')
    serializers.load_npz(args.initmodel, model)
if args.resume:
    logging.info(f'Loading optimizer state from {args.resume}')
    serializers.load_npz(args.resume, optimizer)

logging.info(f'Learning ragte = {args.lr}')

logging.info(f'started reading games from {args.gamelist_train}')

# train data
train_pickle_filename = re.sub(r'\..*?$', '', args.gamelist_train) + '.pickle'
if os.path.exists(train_pickle_filename):
    with open(train_pickle_filename, 'rb') as f:
        positions_train = pickle.load(f)
    logging.info('loading train pickle')
else:
    positions_train = read_kifu(args.gamelist_train)

# test data
test_pickle_filename = re.sub(r'\..*?$', '', args.gamelist_test) + '.pickle'
if os.path.exists(test_pickle_filename):
    with open(test_pickle_filename, 'rb') as f:
        positions_test = pickle.load(f)
    logging.info('loading test pickle')
else:
    positions_test = read_kifu(args.gamelist_test)

# save pickle file if it does not exist
if not os.path.exists(train_pickle_filename):
    with open(train_pickle_filename, 'wb') as f:
        pickle.dump(positions_train, f, 3)
    logging.info('saved train pickle')
if not os.path.exists(test_pickle_filename):
    with open(test_pickle_filename, 'wb') as f:
        pickle.dump(positions_test, f, 3)
    logging.info('saved test pickle')
logging.info('finished reading games')

logging.info(f'train position num = {len(positions_train)}')
logging.info(f'test position num = {len(positions_test)}')


# mini batch
def mini_batch(positions, i, batchsize):
    mini_batch_data = []
    mini_batch_move = []
    mini_batch_win = []
    for b in range(batchsize):
        features, move, win = make_features(positions[i + b])
        mini_batch_data.append(features)
        mini_batch_move.append(move)
        mini_batch_win.append(win)
    return (Variable(cuda.to_gpu(np.array(mini_batch_data, dtype=np.float32))),
            Variable(cuda.to_gpu(np.array(mini_batch_move, dtype=np.int32))),
            Variable(cuda.to_gpu(np.array(mini_batch_win, dtype=np.int32).reshape((-1, 1)))))


def mini_batch_for_test(positions, batchsize):
    mini_batch_data = []
    mini_batch_move = []
    mini_batch_win = []
    for b in range(batchsize):
        features, move, win = make_features(random.choice(positions))
        mini_batch_data.append(features)
        mini_batch_move.append(move)
        mini_batch_win.append(win)

    return (Variable(cuda.to_gpu(np.array(mini_batch_data, dtype=np.float32))),
            Variable(cuda.to_gpu(np.array(mini_batch_move, dtype=np.int32))),
            Variable(cuda.to_gpu(np.array(mini_batch_win, dtype=np.int32).reshape((-1, 1)))))


# train
logging.info('started training')
itr = 0
sum_loss = 0
for e in range(args.epoch):
    positions_train_shuffled = random.sample(positions_train, len(positions_train))

    itr_epoch = 0
    sum_loss_epoch = 0
    for i in range(0, len(positions_train_shuffled) - args.batchsize, args.batchsize):
        x, t1, t2 = mini_batch(positions_train_shuffled, i, args.batchsize)
        y1, y2 = model(x)

        model.cleargrads()
        loss = F.softmax_cross_entropy(y1, t1) + F.sigmoid_cross_entropy(y2, t2)
        loss.backward()
        optimizer.update()

        itr += 1
        sum_loss += loss.data
        itr_epoch += 1
        sum_loss_epoch += loss.data

        # print train loss and test accuracy
        if optimizer.t % args.eval_interval == 0:
            x, t1, t2 = mini_batch_for_test(positions_test, args.test_batchsize)
            y1, y2 = model(x)
            logging.info(
                f'epoch = {optimizer.epoch + 1}, iteration = {optimizer.t}, loss = {sum_loss / itr}, '
                f'accuracy = {F.accuracy(y1, t1).data}, {F.binary_accuracy(y2, t2).data}')
            itr = 0
            sum_loss = 0

    # validating test data
    logging.info('validating test data')
    itr_test = 0
    sum_test_accuracy1 = 0
    sum_test_accuracy2 = 0
    for i in range(0, len(positions_test) - args.batchsize, args.batchsize):
        x, t1, t2 = mini_batch(positions_test, i, args.batchsize)
        y1, y2 = model(x)
        itr_test += 1
        sum_test_accuracy1 += F.accuracy(y1, t1).data
        sum_test_accuracy2 += F.binary_accuracy(y2, t2).data
    logging.info(
        f'epoch = {optimizer.epoch + 1}, iteration = {optimizer.t}, train loss avr = {sum_loss_epoch / itr_epoch}, '
        f'test accuracy = {sum_test_accuracy1 / itr_test}, {sum_test_accuracy2 / itr_test}')
    optimizer.new_epoch()

    logging.info('saving the model')
    serializers.save_npz(args.model, model)
    logging.info('saving the optimizer')
    serializers.save_npz(args.state, optimizer)
