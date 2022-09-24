import argparse
import datetime
import os
import random


def split_array(ar, n_group):
    for i_chunk in range(n_group):
        yield ar[i_chunk * len(ar) // n_group:(i_chunk + 1) * len(ar) // n_group]


parser = argparse.ArgumentParser()

parser.add_argument('dir', type=str)
parser.add_argument('filename', type=str)
parser.add_argument('--ratio', type=float, default=0.9)
parser.add_argument('--split', '-s', type=int, default=1)
args = parser.parse_args()

game_list_all = []
start_time = datetime.datetime.now()
for root, dirs, files in os.walk(args.dir):
    game_list_all.extend([os.path.join(root, file) for file in files])

random.shuffle(game_list_all)

split_list = split_array(game_list_all, args.split)

train_len_all = 0
for i, game_list in enumerate(split_list):

    train_len = int(len(game_list) * args.ratio)
    train_len_all += train_len
    with open(args.filename + f'_{(i + 1)}_train.txt', 'w') as f:
        for j in range(train_len):
            f.write(game_list[j])
            f.write('\n')

    with open(args.filename + f'_{i + 1}_test.txt', 'w') as f:
        for j in range(train_len, len(game_list)):
            f.write(game_list[j])
            f.write('\n')

print(f'total game num = {len(game_list_all)}')
print(f'train game num = {train_len_all}')
print(f'test game num = {len(game_list_all) - train_len_all}')
