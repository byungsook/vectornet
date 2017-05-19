# convert quick, draw! data

import svgwrite
import simplejson as json
import sys
import os.path
import jsonlines

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx

os.chdir('/home/kimby/dev/vectornet')

# output = 'data/qdraw_baseball_64'
# f = open(os.path.join(output,'test.txt'), 'w')
# for root, _, files in os.walk(output):
#     for file in files:
#         if not file.lower().endswith('svg'):
#             continue
#         f.write(file + '\n')
# f.close()

img_size = 128
category = 'cat' # 'baseball' 'stitches' 'cat'
output = 'data/qdraw_{cat}_{img_size}'.format(cat=category, img_size=img_size)

if os.path.exists(output):
    import shutil
    shutil.rmtree(output)
os.mkdir(output)

stroke_width = 2
bbox_pad = 20
cmap = plt.get_cmap('jet')
need = 50

with jsonlines.open('data/{cat}.ndjson'.format(cat=category)) as reader:
    for count, obj in enumerate(reader):
        # print obj
        fn = output + '/' + obj['key_id'] + '.svg'
        if not os.path.isfile(fn):
            print(count, fn)
            dwg = svgwrite.Drawing(fn, profile='tiny', size=(img_size,img_size))
            drawing = obj['drawing']
            num_strokes = len(drawing)
            cnorm = colors.Normalize(vmin=0, vmax=num_strokes-1)
            cscalarmap = cmx.ScalarMappable(norm=cnorm, cmap=cmap)

            # get bbox
            bbox = [100000, 100000, -100000, -100000]
            for i, strokes in enumerate(drawing):
                x = strokes[0]
                y = strokes[1]
                bbox[0] = min(bbox[0], np.amin(x))
                bbox[1] = min(bbox[1], np.amin(y))
                bbox[2] = max(bbox[2], np.amax(x))
                bbox[3] = max(bbox[3], np.amax(y))

            bbox[0] -= bbox_pad
            bbox[1] -= bbox_pad
            bbox[2] += bbox_pad
            bbox[3] += bbox_pad
            # make it square
            dx = bbox[2]-bbox[0]
            dy = bbox[3]-bbox[1]
            b_size = float(max(dx,dy))

            # normalize and save
            for i, strokes in enumerate(drawing):
                x = (np.asarray(strokes[0]) - bbox[0])/b_size*img_size
                y = (np.asarray(strokes[1]) - bbox[1])/b_size*img_size
                # t = strokes[2]
                c = np.asarray(cscalarmap.to_rgba(i))[:3]*255
                c_hex = '#%02x%02x%02x' % (int(c[0]), int(c[1]), int(c[2]))
                dwg.add(dwg.polyline(points=zip(x, y), 
                                     stroke=c_hex,
                                     fill='none',
                                     stroke_width=stroke_width))

            dwg.viewbox(0, 0, img_size, img_size)
            dwg.save()

        # if count >= need:
        #     break

# split dataset
file_list = []
for root, _, files in os.walk(output):
    for file in files:
        file_list.append(file)

num_files = len(file_list)
ids = np.random.permutation(num_files)
train_id = int(num_files * 0.9)
with open(os.path.join(output,'train.txt'), 'w') as f: 
    for id in ids[:train_id]:
        f.write(file_list[id] + '\n')
with open(os.path.join(output,'test.txt'), 'w') as f: 
    for id in ids[train_id:]:
        f.write(file_list[id] + '\n')
