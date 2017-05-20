# qdraw stitches, baseball and cat
python vectorize.py --data_type=qdraw --test_dir=result/no_overlap/qdraw/stitches_128 --data_dir=data/qdraw_stitches_128 --file_list=test.txt --num_test_files=100 --image_height=128 --image_width=128 --batch_size=128 --pathnet_ckpt=pathnet/model/no_trans_128/qdraw_stitches_128/pathnet.ckpt-50000 --find_overlap=False
python postprocess_stat.py --stat_dir=result/no_overlap/qdraw/stitches_128
# python vectorize.py --data_type=qdraw --test_dir=result/overlap/qdraw/stitches_128    --data_dir=data/qdraw_stitches_128 --file_list=test.txt --num_test_files=100 --image_height=128 --image_width=128 --batch_size=128 --pathnet_ckpt=pathnet/model/no_trans_128/qdraw_stitches_128/pathnet.ckpt-50000 --find_overlap=True --ovnet_ckpt=ovnet/model/l2_128/qdraw_stitches_128/ovnet.ckpt-50000
# python postprocess_stat.py --stat_dir=result/overlap/qdraw/stitches_128

python vectorize.py --data_type=qdraw --test_dir=result/no_overlap/qdraw/baseball_128 --data_dir=data/qdraw_baseball_128 --file_list=test.txt --num_test_files=100 --image_height=128 --image_width=128 --batch_size=128 --pathnet_ckpt=pathnet/model/no_trans_128/qdraw_baseball_128/pathnet.ckpt-50000 --find_overlap=False
python postprocess_stat.py --stat_dir=result/no_overlap/qdraw/baseball_128
# python vectorize.py --data_type=qdraw --test_dir=result/overlap/qdraw/baseball_128    --data_dir=data/qdraw_baseball_128 --file_list=test.txt --num_test_files=100 --image_height=128 --image_width=128 --batch_size=128 --pathnet_ckpt=pathnet/model/no_trans_128/qdraw_baseball_128/pathnet.ckpt-50000 --find_overlap=True --ovnet_ckpt=ovnet/model/l2_128/qdraw_baseball_128/ovnet.ckpt-50000
# python postprocess_stat.py --stat_dir=result/overlap/qdraw/baseball_128

python vectorize.py --data_type=qdraw --test_dir=result/no_overlap/qdraw/cat_128 --data_dir=data/qdraw_cat_128 --file_list=test.txt --num_test_files=100 --image_height=128 --image_width=128 --batch_size=128 --pathnet_ckpt=pathnet/model/no_trans_128/qdraw_cat_128/pathnet.ckpt-50000 --find_overlap=False
python postprocess_stat.py --stat_dir=result/no_overlap/qdraw/cat_128
# python vectorize.py --data_type=qdraw --test_dir=result/overlap/qdraw/cat_128    --data_dir=data/qdraw_cat_128 --file_list=test.txt --num_test_files=100 --image_height=128 --image_width=128 --batch_size=128 --pathnet_ckpt=pathnet/model/no_trans_128/qdraw_cat_128/pathnet.ckpt-50000 --find_overlap=True --ovnet_ckpt=ovnet/model/l2_128/qdraw_cat_128/ovnet.ckpt-50000
# python postprocess_stat.py --stat_dir=result/overlap/qdraw/cat_128


# # qdraw test
# python vectorize.py --data_type=qdraw --test_dir=result/no_overlap/qdraw/baseball_64_small --data_dir=data/qdraw_baseball_64_small --file_list=test.txt --num_test_files=100 --image_height=64 --image_width=64 --pathnet_ckpt=pathnet/model/no_trans_64/line/pathnet.ckpt-50000 --find_overlap=False 
# python postprocess_stat.py --stat_dir=result/no_overlap/qdraw/baseball_64_small
# python vectorize.py --data_type=qdraw --test_dir=result/overlap/qdraw/baseball_64_small    --data_dir=data/qdraw_baseball_64_small --file_list=test.txt --num_test_files=100 --image_height=64 --image_width=64 --pathnet_ckpt=pathnet/model/no_trans_64/line/pathnet.ckpt-50000 --find_overlap=True --ovnet_ckpt=ovnet/model/l2_64/line/ovnet.ckpt-50000
# python postprocess_stat.py --stat_dir=result/overlap/qdraw/baseball_64_small

# python vectorize.py --data_type=qdraw --test_dir=result/no_overlap/qdraw_stitches_64_small --data_dir=data/qdraw_stitches_64_small --file_list=test.txt --num_test_files=100 --image_height=64 --image_width=64 --pathnet_ckpt=pathnet/model/no_trans_64/line/pathnet.ckpt-50000 --find_overlap=False 
# python postprocess_stat.py --stat_dir=result/no_overlap/qdraw_stitches_64_small
# python vectorize.py --data_type=qdraw --test_dir=result/overlap/qdraw_stitches_64_small    --data_dir=data/qdraw_stitches_64_small --file_list=test.txt --num_test_files=100 --image_height=64 --image_width=64 --pathnet_ckpt=pathnet/model/no_trans_64/line/pathnet.ckpt-50000 --find_overlap=True --ovnet_ckpt=ovnet/model/l2_64/line/ovnet.ckpt-50000
# python postprocess_stat.py --stat_dir=result/overlap/qdraw_stitches_64_small


# ovnet: l2 loss
# python vectorize.py --data_type=chinese --test_dir=result/overlap/ch1_l2  --num_test_files=100 --image_height=64 --image_width=64 --pathnet_ckpt=pathnet/model/no_trans_64/ch1/pathnet.ckpt-50000  --ovnet_ckpt=ovnet/model/l2_64/ch1/ovnet.ckpt-50000  --find_overlap=True --data_dir=data/chinese1 --chinese1=True --file_list=test.txt
# python postprocess_stat.py --stat_dir=result/overlap/ch1_l2
# python vectorize.py --data_type=chinese --test_dir=result/overlap/ch2_l2  --num_test_files=100 --image_height=64 --image_width=64 --pathnet_ckpt=pathnet/model/no_trans_64/ch2/pathnet.ckpt-50000  --ovnet_ckpt=ovnet/model/l2_64/ch2/ovnet.ckpt-50000  --find_overlap=True --data_dir=data/chinese2 --chinese1=False --file_list=test.txt
# python postprocess_stat.py --stat_dir=result/overlap/ch2_l2
# python vectorize.py --data_type=line --test_dir=result/overlap/line_l2 --num_test_files=100 --image_height=64 --image_width=64 --pathnet_ckpt=pathnet/model/no_trans_64/line/pathnet.ckpt-50000 --ovnet_ckpt=ovnet/model/l2_64/line/ovnet.ckpt-50000 --find_overlap=True --data_dir=data/line_ov --file_list=
# python postprocess_stat.py --stat_dir=result/overlap/line_l2

# python vectorize.py --data_type=chinese --test_dir=result/no_overlap/ch1  --num_test_files=100 --image_height=64 --image_width=64 --pathnet_ckpt=pathnet/model/no_trans_64/ch1/pathnet.ckpt-50000  --find_overlap=False --data_dir=data/chinese1 --chinese1=True --file_list=test.txt
# python postprocess_stat.py --stat_dir=result/no_overlap/ch1
# python vectorize.py --data_type=chinese --test_dir=result/no_overlap/ch2  --num_test_files=100 --image_height=64 --image_width=64 --pathnet_ckpt=pathnet/model/no_trans_64/ch2/pathnet.ckpt-50000  --find_overlap=False --data_dir=data/chinese2 --chinese1=False --file_list=test.txt
# python postprocess_stat.py --stat_dir=result/no_overlap/ch2
# python vectorize.py --data_type=line --test_dir=result/no_overlap/line  --num_test_files=100 --image_height=64 --image_width=64 --pathnet_ckpt=pathnet/model/no_trans_64/line/pathnet.ckpt-50000  --find_overlap=False --data_dir=data/line_ov --file_list=
# python postprocess_stat.py --stat_dir=result/no_overlap/line


# # ch1 test again
# python vectorize.py --data_type=chinese --test_dir=result/overlap_gco/ch1_    --num_test_files=100 --image_height=64 --image_width=64 --pathnet_ckpt=pathnet/model/no_trans_64/ch1/pathnet.ckpt-50000    --ovnet_ckpt=ovnet/model/no_trans_64/ch1/ovnet.ckpt-50000   --find_overlap=True --data_dir=data/chinese1 --chinese1=True
# python postprocess_stat.py --stat_dir=result/overlap_gco/ch1_

# # sketch
# python vectorize.py --data_type=sketch --test_dir=result/no_overlap/bicycle_rl128  --pathnet_ckpt=pathnet/model/no_trans_128/line/pathnet.ckpt-50000 --data_dir=data/bicycle --num_test_files=100 --image_width=128 --image_height=96 --batch_size=128 --find_overlap=False
# python postprocess_stat.py --stat_dir=result/no_overlap/bicycle_rl128

# python vectorize.py --data_type=sketch --test_dir=result/no_overlap/bicycle_rl  --pathnet_ckpt=pathnet/model/no_trans_64/line/pathnet.ckpt-50000 --data_dir=data/bicycle --num_test_files=100 --image_width=128 --image_height=96 --batch_size=128 --find_overlap=False
# python postprocess_stat.py --stat_dir=result/no_overlap/bicycle_rl
# python vectorize.py --data_type=sketch --test_dir=result/overlap_gco/bicycle_rl --pathnet_ckpt=pathnet/model/no_trans_64/line/pathnet.ckpt-50000 --data_dir=data/bicycle --num_test_files=100 --image_width=128 --image_height=96 --batch_size=128 --find_overlap=True  --ovnet_ckpt=ovnet/model/no_trans_64/line/ovnet.ckpt-50000
# python postprocess_stat.py --stat_dir=result/overlap_gco/bicycle_rl

# python vectorize.py --data_type=sketch --test_dir=result/no_overlap/snail_rl  --pathnet_ckpt=pathnet/model/no_trans_64/line/pathnet.ckpt-50000 --data_dir=data/snail --num_test_files=100 --image_width=128 --image_height=96 --batch_size=128 --find_overlap=False
# python postprocess_stat.py --stat_dir=result/no_overlap/snail_rl
# python vectorize.py --data_type=sketch --test_dir=result/overlap_gco/snail_rl --pathnet_ckpt=pathnet/model/no_trans_64/line/pathnet.ckpt-50000 --data_dir=data/snail --num_test_files=100 --image_width=128 --image_height=96 --batch_size=128 --find_overlap=True  --ovnet_ckpt=ovnet/model/no_trans_64/line/ovnet.ckpt-50000
# python postprocess_stat.py --stat_dir=result/overlap_gco/snail_rl


# python vectorize.py --data_type=sketch --test_dir=result/no_overlap/bicycle  --pathnet_ckpt=pathnet/model/no_trans_128/bicycle/pathnet.ckpt-50000 --data_dir=data/bicycle --num_test_files=100 --image_width=128 --image_height=96 --batch_size=128 --find_overlap=False
# python postprocess_stat.py --stat_dir=result/no_overlap/bicycle
# python vectorize.py --data_type=sketch --test_dir=result/overlap_gco/bicycle --pathnet_ckpt=pathnet/model/no_trans_128/bicycle/pathnet.ckpt-50000 --data_dir=data/bicycle --num_test_files=100 --image_width=128 --image_height=96 --batch_size=128 --find_overlap=True  --ovnet_ckpt=ovnet/model/no_trans_128/bicycle/ovnet.ckpt-50000
# python postprocess_stat.py --stat_dir=result/overlap_gco/bicycle

# python vectorize.py --data_type=sketch --test_dir=result/no_overlap/car  --pathnet_ckpt=pathnet/model/no_trans_128/car/pathnet.ckpt-45000 --data_dir=data/car --num_test_files=100 --image_width=128 --image_height=96 --batch_size=128 --find_overlap=False
# python postprocess_stat.py --stat_dir=result/no_overlap/car
# python vectorize.py --data_type=sketch --test_dir=result/overlap_gco/car --pathnet_ckpt=pathnet/model/no_trans_128/car/pathnet.ckpt-45000 --data_dir=data/car --num_test_files=100 --image_width=128 --image_height=96 --batch_size=128 --find_overlap=True  --ovnet_ckpt=ovnet/model/no_trans_128/car/ovnet.ckpt-50000
# python postprocess_stat.py --stat_dir=result/overlap_gco/car

# python vectorize.py --data_type=sketch --test_dir=result/no_overlap/snail  --pathnet_ckpt=pathnet/model/no_trans_128/snail/pathnet.ckpt-50000 --data_dir=data/snail --num_test_files=100 --image_width=128 --image_height=96 --batch_size=128 --find_overlap=False
# python postprocess_stat.py --stat_dir=result/no_overlap/snail
# python vectorize.py --data_type=sketch --test_dir=result/overlap_gco/snail --pathnet_ckpt=pathnet/model/no_trans_128/snail/pathnet.ckpt-50000 --data_dir=data/snail --num_test_files=100 --image_width=128 --image_height=96 --batch_size=128 --find_overlap=True  --ovnet_ckpt=ovnet/model/no_trans_128/snail/ovnet.ckpt-50000
# python postprocess_stat.py --stat_dir=result/overlap_gco/snail


# python vectorize.py --data_type=sketch --test_dir=result/no_overlap/bicycle_tr  --num_test_files=100 --image_width=128 --image_height=96 --pathnet_ckpt=pathnet/model/no_trans_128/bicycle/pathnet.ckpt-50000  --find_overlap=False --data_dir=data/bicycle --batch_size=128 --file_list=train.txt
# python vectorize.py --data_type=sketch --test_dir=result/overlap_gco/bicycle_tr --num_test_files=100 --image_width=128 --image_height=96 --pathnet_ckpt=pathnet/model/no_trans_128/bicycle/pathnet.ckpt-50000   --ovnet_ckpt=ovnet/model/no_trans_128/bicycle/ovnet.ckpt-50000   --find_overlap=True  --data_dir=data/bicycle --batch_size=128 --file_list=train.txt
# python postprocess_stat.py --stat_dir=result/no_overlap/bicycle_tr
# python postprocess_stat.py --stat_dir=result/overlap_gco/bicycle_tr

# python vectorize.py --data_type=sketch --test_dir=result/no_overlap/bicycle  --num_test_files=100 --image_width=128 --image_height=96 --pathnet_ckpt=pathnet/model/no_trans_128/bicycle/pathnet.ckpt-50000   --find_overlap=False --data_dir=data/bicycle --batch_size=128
# python vectorize.py --data_type=sketch2 --test_dir=result/overlap_gco/bicycle --num_test_files=100 --image_width=128 --image_height=96 --pathnet_ckpt=pathnet/model/no_trans_128/bicycle/pathnet.ckpt-50000   --ovnet_ckpt=ovnet/model/no_trans_128/bibycle/ovnet.ckpt-50000   --find_overlap=True  --data_dir=data/bicycle --batch_size=128
# python postprocess_stat.py --stat_dir=result/no_overlap/bicycle
# python postprocess_stat.py --stat_dir=result/overlap_gco/bicycle


# # sketch2_l
# python vectorize.py --data_type=sketch2 --test_dir=log/no_overlap/sketch2_l_train  --num_test_files=100 --image_height=128 --image_width=128 --pathnet_ckpt=pathnet/model/no_trans_128/sketch2_l/pathnet.ckpt-50000   --ovnet_ckpt=ovnet/model/no_trans_128/sketch2_l/ovnet.ckpt-50000   --find_overlap=False --data_dir=data/sketch_schneider_l --batch_size=128 --file_list=train.txt --max_num_labels=16
# python vectorize.py --data_type=sketch2 --test_dir=log/overlap_gco/sketch2_l_train --num_test_files=100 --image_height=128 --image_width=128 --pathnet_ckpt=pathnet/model/no_trans_128/sketch2_l/pathnet.ckpt-50000   --ovnet_ckpt=ovnet/model/no_trans_128/sketch2_l/ovnet.ckpt-50000   --find_overlap=True  --data_dir=data/sketch_schneider_l --batch_size=128 --file_list=train.txt --max_num_labels=16
# python postprocess_stat.py --stat_dir=log/no_overlap/sketch2_l_train
# python postprocess_stat.py --stat_dir=log/overlap_gco/sketch2_l_train

# python vectorize.py --data_type=sketch2 --test_dir=log/no_overlap/sketch2_l  --num_test_files=100 --image_height=128 --image_width=128 --pathnet_ckpt=pathnet/model/no_trans_128/sketch2_l/pathnet.ckpt-50000   --ovnet_ckpt=ovnet/model/no_trans_128/sketch2_l/ovnet.ckpt-50000   --find_overlap=False --data_dir=data/sketch_schneider_l --batch_size=128 --max_num_labels=16
# python vectorize.py --data_type=sketch2 --test_dir=log/overlap_gco/sketch2_l --num_test_files=100 --image_height=128 --image_width=128 --pathnet_ckpt=pathnet/model/no_trans_128/sketch2_l/pathnet.ckpt-50000   --ovnet_ckpt=ovnet/model/no_trans_128/sketch2_l/ovnet.ckpt-50000   --find_overlap=True  --data_dir=data/sketch_schneider_l --batch_size=128 --max_num_labels=16
# python postprocess_stat.py --stat_dir=log/no_overlap/sketch2_l
# python postprocess_stat.py --stat_dir=log/overlap_gco/sketch2_l

# sketch2
# python vectorize.py --data_type=sketch2 --test_dir=log/no_overlap/sketch2_train  --num_test_files=100 --image_height=128 --image_width=128 --pathnet_ckpt=pathnet/model/no_trans_128/sketch2/pathnet.ckpt-50000   --ovnet_ckpt=ovnet/model/no_trans_128/sketch2/ovnet.ckpt-50000   --find_overlap=False --data_dir=data/sketch_schneider --batch_size=128 --file_list=train.txt
# python vectorize.py --data_type=sketch2 --test_dir=log/overlap_gco/sketch2_train --num_test_files=100 --image_height=128 --image_width=128 --pathnet_ckpt=pathnet/model/no_trans_128/sketch2/pathnet.ckpt-50000   --ovnet_ckpt=ovnet/model/no_trans_128/sketch2/ovnet.ckpt-50000   --find_overlap=True  --data_dir=data/sketch_schneider --batch_size=128 --file_list=train.txt
# python postprocess_stat.py --stat_dir=log/no_overlap/sketch2_train
# python postprocess_stat.py --stat_dir=log/overlap_gco/sketch2_train

# python vectorize.py --data_type=sketch2 --test_dir=log/no_overlap/sketch2  --num_test_files=100 --image_height=128 --image_width=128 --pathnet_ckpt=pathnet/model/no_trans_128/sketch2/pathnet.ckpt-50000   --ovnet_ckpt=ovnet/model/no_trans_128/sketch2/ovnet.ckpt-50000   --find_overlap=False --data_dir=data/sketch_schneider --batch_size=128
# python vectorize.py --data_type=sketch2 --test_dir=log/overlap_gco/sketch2 --num_test_files=100 --image_height=128 --image_width=128 --pathnet_ckpt=pathnet/model/no_trans_128/sketch2/pathnet.ckpt-50000   --ovnet_ckpt=ovnet/model/no_trans_128/sketch2/ovnet.ckpt-50000   --find_overlap=True  --data_dir=data/sketch_schneider --batch_size=128
# python postprocess_stat.py --stat_dir=log/no_overlap/sketch2
# python postprocess_stat.py --stat_dir=log/overlap_gco/sketch2

# 24-03-17 Fri. max label test (default: 64, test: 23, 32, 128 on ch1) --> 128 is the best
# python vectorize.py --data_type=chinese --test_dir=log/no_overlap_m23/ch1    --num_test_files=100 --image_height=64 --image_width=64 --pathnet_ckpt=pathnet/model/no_trans_64/ch1/pathnet.ckpt-50000    --ovnet_ckpt=ovnet/model/no_trans_64/ch1/ovnet.ckpt-50000   --find_overlap=False --max_num_labels=23  --data_dir=data/chinese1 --chinese1=True
# python vectorize.py --data_type=chinese --test_dir=log/no_overlap_m32/ch1    --num_test_files=100 --image_height=64 --image_width=64 --pathnet_ckpt=pathnet/model/no_trans_64/ch1/pathnet.ckpt-50000    --ovnet_ckpt=ovnet/model/no_trans_64/ch1/ovnet.ckpt-50000   --find_overlap=False --max_num_labels=32  --data_dir=data/chinese1 --chinese1=True
# python vectorize.py --data_type=chinese --test_dir=log/no_overlap_m128/ch1   --num_test_files=100 --image_height=64 --image_width=64 --pathnet_ckpt=pathnet/model/no_trans_64/ch1/pathnet.ckpt-50000    --ovnet_ckpt=ovnet/model/no_trans_64/ch1/ovnet.ckpt-50000   --find_overlap=False --max_num_labels=128 --data_dir=data/chinese1 --chinese1=True
# python vectorize.py --data_type=chinese --test_dir=log/no_overlap_m256/ch1   --num_test_files=100 --image_height=64 --image_width=64 --pathnet_ckpt=pathnet/model/no_trans_64/ch1/pathnet.ckpt-50000    --ovnet_ckpt=ovnet/model/no_trans_64/ch1/ovnet.ckpt-50000   --find_overlap=False --max_num_labels=256 --data_dir=data/chinese1 --chinese1=True

# python postprocess_stat.py --stat_dir=log/no_overlap_m23/ch1
# python postprocess_stat.py --stat_dir=log/no_overlap_m32/ch1
# python postprocess_stat.py --stat_dir=log/no_overlap_m128/ch1

# python vectorize.py --data_type=chinese --test_dir=log/overlap_gco_m23/ch1    --num_test_files=100 --image_height=64 --image_width=64 --pathnet_ckpt=pathnet/model/no_trans_64/ch1/pathnet.ckpt-50000    --ovnet_ckpt=ovnet/model/no_trans_64/ch1/ovnet.ckpt-50000   --find_overlap=True --max_num_labels=23  --data_dir=data/chinese1 --chinese1=True
# python vectorize.py --data_type=chinese --test_dir=log/overlap_gco_m32/ch1    --num_test_files=100 --image_height=64 --image_width=64 --pathnet_ckpt=pathnet/model/no_trans_64/ch1/pathnet.ckpt-50000    --ovnet_ckpt=ovnet/model/no_trans_64/ch1/ovnet.ckpt-50000   --find_overlap=True --max_num_labels=32  --data_dir=data/chinese1 --chinese1=True
# python vectorize.py --data_type=chinese --test_dir=log/overlap_gco_m128/ch1   --num_test_files=100 --image_height=64 --image_width=64 --pathnet_ckpt=pathnet/model/no_trans_64/ch1/pathnet.ckpt-50000    --ovnet_ckpt=ovnet/model/no_trans_64/ch1/ovnet.ckpt-50000   --find_overlap=True --max_num_labels=128 --data_dir=data/chinese1 --chinese1=True
# python vectorize.py --data_type=chinese --test_dir=log/overlap_gco_m256/ch1   --num_test_files=100 --image_height=64 --image_width=64 --pathnet_ckpt=pathnet/model/no_trans_64/ch1/pathnet.ckpt-50000    --ovnet_ckpt=ovnet/model/no_trans_64/ch1/ovnet.ckpt-50000   --find_overlap=True --max_num_labels=256 --data_dir=data/chinese1 --chinese1=True

# python postprocess_stat.py --stat_dir=log/overlap_gco_m23/ch1
# python postprocess_stat.py --stat_dir=log/overlap_gco_m32/ch1
# python postprocess_stat.py --stat_dir=log/overlap_gco_m128/ch1


# 22-03-17 Wed. run without overlaps
# python vectorize.py --data_type=chinese --test_dir=log/no_overlap/ch1    --num_test_files=100 --image_height=64 --image_width=64 --pathnet_ckpt=pathnet/model/no_trans_64/ch1/pathnet.ckpt-50000    --ovnet_ckpt=ovnet/model/no_trans_64/ch1/ovnet.ckpt-50000   --find_overlap=False --data_dir=data/chinese1 --chinese1=True
# python vectorize.py --data_type=chinese --test_dir=log/no_overlap/ch2    --num_test_files=100 --image_height=64 --image_width=64 --pathnet_ckpt=pathnet/model/no_trans_64/ch2/pathnet.ckpt-50000    --ovnet_ckpt=ovnet/model/no_trans_64/ch2/ovnet.ckpt-50000   --find_overlap=False --data_dir=data/chinese2 --chinese1=False
# python vectorize.py --data_type=line    --test_dir=log/no_overlap/line   --num_test_files=100 --image_height=64 --image_width=64 --pathnet_ckpt=pathnet/model/no_trans_64/line/pathnet.ckpt-50000   --ovnet_ckpt=ovnet/model/no_trans_64/line/ovnet.ckpt-50000  --find_overlap=False --max_stroke_width=2 --num_paths=4

# python postprocess_stat.py --stat_dir=log/no_overlap/ch1
# python postprocess_stat.py --stat_dir=log/no_overlap/ch2
# python postprocess_stat.py --stat_dir=log/no_overlap/line

# # no overlap with 0 penalty --> seems doesn't work for overlap
# # python vectorize.py --data_type=chinese --test_dir=log/overlap_0/ch1    --num_test_files=100 --image_height=64 --image_width=64 --pathnet_ckpt=pathnet/model/no_trans_64/ch1/pathnet.ckpt-50000    --ovnet_ckpt=ovnet/model/no_trans_64/ch1/ovnet.ckpt-50000   --find_overlap=True --data_dir=data/chinese1 --chinese1=True
# # python postprocess_stat.py --stat_dir=log/overlap_0/ch1

# # no overlap with qpbo-iter1, iter3 --> seems doesn't work for overlap...

# # -1000 penalty --> works, but invalid

# w/ overlaps, strong spatial weight for overlaped pixel pair
# python vectorize.py --data_type=chinese --test_dir=log/overlap_gco/ch1    --num_test_files=100 --image_height=64 --image_width=64 --pathnet_ckpt=pathnet/model/no_trans_64/ch1/pathnet.ckpt-50000    --ovnet_ckpt=ovnet/model/no_trans_64/ch1/ovnet.ckpt-50000   --find_overlap=True --data_dir=data/chinese1 --chinese1=True
# python vectorize.py --data_type=chinese --test_dir=log/overlap_gco/ch2    --num_test_files=100 --image_height=64 --image_width=64 --pathnet_ckpt=pathnet/model/no_trans_64/ch2/pathnet.ckpt-50000    --ovnet_ckpt=ovnet/model/no_trans_64/ch2/ovnet.ckpt-50000   --find_overlap=True --data_dir=data/chinese2 --chinese1=False
# python vectorize.py --data_type=line    --test_dir=log/overlap_gco/line   --num_test_files=100 --image_height=64 --image_width=64 --pathnet_ckpt=pathnet/model/no_trans_64/line/pathnet.ckpt-50000   --ovnet_ckpt=ovnet/model/no_trans_64/line/ovnet.ckpt-50000  --find_overlap=True --max_stroke_width=2 --num_paths=4

# python postprocess_stat.py --stat_dir=log/overlap_gco/ch1
# python postprocess_stat.py --stat_dir=log/overlap_gco/ch2
# python postprocess_stat.py --stat_dir=log/overlap_gco/line

# use spatial sigma 10 instead of 8 --> doesn't work well
# python vectorize.py --data_type=chinese --test_dir=log/overlap_gco_n10/ch1    --num_test_files=100 --image_height=64 --image_width=64 --pathnet_ckpt=pathnet/model/no_trans_64/ch1/pathnet.ckpt-50000    --ovnet_ckpt=ovnet/model/no_trans_64/ch1/ovnet.ckpt-50000   --find_overlap=True --neighbor_sigma=10 --data_dir=data/chinese1 --chinese1=True
# python vectorize.py --data_type=chinese --test_dir=log/overlap_gco_n10/ch2    --num_test_files=100 --image_height=64 --image_width=64 --pathnet_ckpt=pathnet/model/no_trans_64/ch2/pathnet.ckpt-50000    --ovnet_ckpt=ovnet/model/no_trans_64/ch2/ovnet.ckpt-50000   --find_overlap=True --neighbor_sigma=10 --data_dir=data/chinese2 --chinese1=False
# python vectorize.py --data_type=line    --test_dir=log/overlap_gco_n10/line   --num_test_files=100 --image_height=64 --image_width=64 --pathnet_ckpt=pathnet/model/no_trans_64/line/pathnet.ckpt-50000   --ovnet_ckpt=ovnet/model/no_trans_64/line/ovnet.ckpt-50000  --find_overlap=True --neighbor_sigma=10 --max_stroke_width=2 --num_paths=4

# python postprocess_stat.py --stat_dir=log/overlap_gco_n10/ch1
# python postprocess_stat.py --stat_dir=log/overlap_gco_n10/ch2
# python postprocess_stat.py --stat_dir=log/overlap_gco_n10/line


# # trans-trans
# python vectorize.py --data_type=chinese --test_dir=test/ch1    --num_test_files=100 --image_height=48 --image_width=48 --pathnet_ckpt=pathnet/model/trans/ch1/pathnet.ckpt    --ovnet_ckpt=ovnet/model/trans/ch1/ovnet.ckpt    --data_dir=data/chinese1 --chinese1=True
# python vectorize.py --data_type=chinese --test_dir=test/ch2    --num_test_files=100 --image_height=48 --image_width=48 --pathnet_ckpt=pathnet/model/trans/ch2/pathnet.ckpt    --ovnet_ckpt=ovnet/model/trans/ch2/ovnet.ckpt    --data_dir=data/chinese2 --chinese1=False
# python vectorize.py --data_type=line    --test_dir=test/line   --num_test_files=100 --image_height=48 --image_width=48 --pathnet_ckpt=pathnet/model/no_trans/line/pathnet.ckpt   --ovnet_ckpt=ovnet/model/no_trans/line/ovnet.ckpt
# python vectorize.py --data_type=sketch  --test_dir=test/sketch --num_test_files=100 --image_height=64 --image_width=48 --pathnet_ckpt=pathnet/model/trans/sketch/pathnet.ckpt --ovnet_ckpt=ovnet/model/trans/sketch/ovnet.ckpt --data_dir=data/sketch --max_num_labels=256

# no trans-no trans: useless...
