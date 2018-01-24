python main.py --is_train=False --dataset=ch --load_pathnet=log/path/ch_1231_170036_win --find_overlap=False --tag=nv
python main.py --is_train=False --dataset=ch --load_pathnet=log/path/ch_1231_170036_win --load_overlapnet=log/overlap/ch_0101_012450_win --tag=ov

python main.py --is_train=False --dataset=kanji --load_pathnet=log/path/kanji_1231_175226_win --find_overlap=False --tag=nv
python main.py --is_train=False --dataset=kanji --load_pathnet=log/path/kanji_1231_175226_win --load_overlapnet=log/overlap/kanji_0101_035424_win --tag=ov

python main.py --is_train=False --dataset=line --load_pathnet=log/path/line_1231_162901_win --find_overlap=False --tag=nv
python main.py --is_train=False --dataset=line --load_pathnet=log/path/line_1231_162901_win --load_overlapnet=log/overlap/line_0101_003631_win --tag=ov

python main.py --is_train=False --dataset=baseball --load_pathnet=log/path/baseball_1231_185540_win --find_overlap=False --tag=nv --height=128 --width=128 --neighbor_sample=0.02 --test_batch_size=256
python main.py --is_train=False --dataset=baseball --load_pathnet=log/path/baseball_1231_185540_win --load_overlapnet=log/overlap/baseball_0101_084611_win --tag=ov --height=128 --width=128 --neighbor_sample=0.02 --test_batch_size=256

python main.py --is_train=False --dataset=cat --load_pathnet=log/path/cat_1231_205036_win --find_overlap=False --tag=nv --height=128 --width=128 --neighbor_sample=0.02 --test_batch_size=256
python main.py --is_train=False --dataset=cat --load_pathnet=log/path/cat_1231_205036_win --load_overlapnet=log/overlap/cat_0101_104304_win --tag=ov --height=128 --width=128 --neighbor_sample=0.02 --test_batch_size=256