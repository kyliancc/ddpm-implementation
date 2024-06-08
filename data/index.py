import os
import random
import json


file_list = os.listdir('./animefacedataset/images')
random.shuffle(file_list)

nimages = len(file_list)

nvals = int(nimages * 0.1)
ntrains = nimages - nvals

train_imgs = file_list[:ntrains]
val_imgs = file_list[ntrains:]

train_data = [{'file': train_imgs[i]} for i in range(ntrains)]
val_data = [{'file': val_imgs[i]} for i in range(nvals)]

train_json = json.dumps(train_data, sort_keys=True, indent=4, separators=(', ', ': '))
with open('./train-index.json', 'w') as f:
    f.write(train_json)
    print('Saved train-index.json.')
val_json = json.dumps(val_data, sort_keys=True, indent=4, separators=(', ', ': '))
with open('./val-index.json', 'w') as f:
    f.write(val_json)
    print('Saved val-index.json.')
