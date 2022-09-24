import argparse
import os
import re
import statistics

parser = argparse.ArgumentParser()
parser.add_argument('dir', type=str)
args = parser.parse_args()


def find_all_files(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            yield os.path.join(root, file)


ptn_rate = re.compile(r"^'(black|white)_rate:.*:(.*)$")

game_count = 0
rates = []

print(f'filtering {args.dir}/')

for filepath in find_all_files(args.dir):
    rate = {}
    move_len = 0
    has_resigned = False

    for line in open(filepath, 'r', encoding='utf-8'):
        line = line.strip()
        m = ptn_rate.match(line)
        if m:
            rate[m.group(1)] = float(m.group(2))
        if line[:1] == '+' or line[:1] == '-':
            move_len += 1
        if line == '%TORYO':
            has_resigned = True

    if not has_resigned or move_len <= 50 or len(rate) < 2 or min(rate.values()) < 3500:
        os.remove(filepath)
    else:
        game_count += 1
        rates.extend([_ for _ in rate.values()])

print('games count:', game_count)
print(f'rate mean: {statistics.mean(rates)}')
print(f'rate median: {statistics.median(rates)}')
print(f'rate max: {max(rates)}')
print(f'rate min: {min(rates)}')
