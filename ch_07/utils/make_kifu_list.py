import argparse
import os
import random

parser = argparse.ArgumentParser()

parser.add_argument('dir', type=str)
parser.add_argument('filename', type=str)
parser.add_argument('--ratio', type=float, default=0.9)
args = parser.parse_args()

kifu_list = []
for root, dirs, files in os.walk(args.dir):
    for file in files:
        kifu_list.append(os.path.join(root, file))

random.shuffle(kifu_list)

train_len = int(len(kifu_list) * args.ratio)
with open(args.filename + '_train.txt', 'w') as f:
    for i in range(train_len):
        f.write(kifu_list[i])
        f.write('\n')

with open(args.filename + '_test.txt', 'w') as f:
    for i in range(train_len, len(kifu_list)):
        f.write(kifu_list[i])
        f.write('\n')

print(f'total kifu num = {len(kifu_list)}')
print(f'train kifu num = {train_len}')
print(f'test kifu num = {len(kifu_list) - train_len}')
