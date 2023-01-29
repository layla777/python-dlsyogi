from chainer import serializers

from pydlshogi.network.policy_bn import *
from pydlshogi.network.value_bn import *

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('policy_model', type=str)
parser.add_argument('value_model', type=str)

args = parser.parse_args()

policy_model = PolicyNetwork()
value_model = ValueNetwork()

print('Loading policy model from', args.policy_model)
serializers.load_npz(args.policy_model, policy_model)

print('value model params')
value_dict = {}
for path, param in value_model.namedparams():
    print(path, param.data.shape)
    value_dict[path] = param

print('policy model params')
for path, param in policy_model.namedparams():
    print(path, param.data.shape)
    if path in value_dict:
        value_dict[path].data = param.data

print('saving the model')
serializers.save_npz(args.value_model, value_model)