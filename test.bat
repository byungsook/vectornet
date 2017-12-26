
python main.py --is_train=False --dataset=cat --load_pathnet=log/path/cat_1224_103053_01 --load_overlapnet=log/overlap/cat_1225_011822_01 --tag=ov --height=128 --width=128 --neighbor_sample=0.02 --test_batch_size=256

python main.py --is_train=False --dataset=multi --load_pathnet=log/path/multi_1224_124657_01 --find_overlap=False --tag=nv --height=128 --width=128 --neighbor_sample=0.02 --test_batch_size=256
python main.py --is_train=False --dataset=multi --load_pathnet=log/path/multi_1224_124657_01 --load_overlapnet=log/overlap/multi_1225_041801_01 --tag=ov --height=128 --width=128 --neighbor_sample=0.02 --test_batch_size=256


REM python main.py --is_train=False --dataset=line --load_pathnet=log/path/line_1222_095616_test --load_overlapnet=log/overlap/line_1222_105045_test
