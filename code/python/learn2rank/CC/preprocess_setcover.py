import pathlib

sizes = ['100_3', '100_4', '100_5', '100_6', '100_7',
         '150_3', '150_4']
case = 1
n_items = {'train': (0, 1000),
           'val': (1000, 1100),
           'test': (1100, 1200)}

text = ''
for size in sizes:
    for split in ['train', 'val', 'test']:
        start, end = n_items.get(split)

        for pid in range(start, end, 100):
            text += f'{case} ' \
                    f'python -m learn2rank.scripts.preprocess_setcover ' \
                    f'--machine cc ' \
                    f'--size {size} ' \
                    f'--split {split} ' \
                    f'--from_pid {pid} ' \
                    f'--to_pid {pid + 100}\n'

            case += 1

pathlib.Path('./table.dat').open('w').write(text)
