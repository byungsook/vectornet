

python main.py --is_train=False --dataset=ch --load_pathnet=log/path/ch_1223_170630_01 --find_overlap=False --tag=nv
python main.py --is_train=False --dataset=ch --load_pathnet=log/path/ch_1223_170630_01 --load_overlapnet=log/overlap/ch_1224_160412_01 --tag=ov

python main.py --is_train=False --dataset=kanji --load_pathnet=log/path/kanji_1223_175926_01 --find_overlap=False --tag=nv
python main.py --is_train=False --dataset=kanji --load_pathnet=log/path/kanji_1223_175926_01 --load_overlapnet=log/overlap/kanji_1224_183116_01 --tag=ov

python main.py --is_train=False --dataset=line --load_pathnet=log/path/line_1224_152055_01 --find_overlap=False --tag=nv
python main.py --is_train=False --dataset=line --load_pathnet=log/path/line_1224_152055_01 --load_overlapnet=log/overlap/line_1225_060657_01 --tag=ov

python main.py --is_train=False --dataset=baseball --load_pathnet=log/path/baseball_1224_081511_01 --find_overlap=False --tag=nv --height=128 --width=128 --neighbor_sample=0.02 --test_batch_size=256
python main.py --is_train=False --dataset=baseball --load_pathnet=log/path/baseball_1224_081511_01 --load_overlapnet=log/overlap/baseball_1224_224311_01 --tag=ov --height=128 --width=128 --neighbor_sample=0.02 --test_batch_size=256

python main.py --is_train=False --dataset=cat --load_pathnet=log/path/cat_1224_103053_01 --find_overlap=False --tag=nv --height=128 --width=128 --neighbor_sample=0.02 --test_batch_size=256
python main.py --is_train=False --dataset=cat --load_pathnet=log/path/cat_1224_103053_01 --load_overlapnet=log/overlap/cat_1225_011822_01 --tag=ov --height=128 --width=128 --neighbor_sample=0.02 --test_batch_size=256

python main.py --is_train=False --dataset=multi --load_pathnet=log/path/multi_1224_124657_01 --find_overlap=False --tag=nv --height=128 --width=128 --neighbor_sample=0.02 --test_batch_size=256
python main.py --is_train=False --dataset=multi --load_pathnet=log/path/multi_1224_124657_01 --load_overlapnet=log/overlap/multi_1225_041801_01 --tag=ov --height=128 --width=128 --neighbor_sample=0.02 --test_batch_size=256