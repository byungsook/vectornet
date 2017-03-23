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

# # no overlap with qpbo-iter1 --> seems doesn't work for overlap...

# # invalid because of negative edge weight, but just in case..
# python vectorize.py --data_type=chinese --test_dir=log/overlap_gco/ch1    --num_test_files=100 --image_height=64 --image_width=64 --pathnet_ckpt=pathnet/model/no_trans_64/ch1/pathnet.ckpt-50000    --ovnet_ckpt=ovnet/model/no_trans_64/ch1/ovnet.ckpt-50000   --find_overlap=True --data_dir=data/chinese1 --chinese1=True
# python vectorize.py --data_type=chinese --test_dir=log/overlap_gco/ch2    --num_test_files=100 --image_height=64 --image_width=64 --pathnet_ckpt=pathnet/model/no_trans_64/ch2/pathnet.ckpt-50000    --ovnet_ckpt=ovnet/model/no_trans_64/ch2/ovnet.ckpt-50000   --find_overlap=True --data_dir=data/chinese2 --chinese1=False
# python vectorize.py --data_type=line    --test_dir=log/overlap_gco/line   --num_test_files=100 --image_height=64 --image_width=64 --pathnet_ckpt=pathnet/model/no_trans_64/line/pathnet.ckpt-50000   --ovnet_ckpt=ovnet/model/no_trans_64/line/ovnet.ckpt-50000  --find_overlap=True --max_stroke_width=2 --num_paths=4

# python postprocess_stat.py --stat_dir=log/overlap_gco/ch1
# python postprocess_stat.py --stat_dir=log/overlap_gco/ch2
# python postprocess_stat.py --stat_dir=log/overlap_gco/line



# # trans-trans
# python vectorize.py --data_type=chinese --test_dir=test/ch1    --num_test_files=100 --image_height=48 --image_width=48 --pathnet_ckpt=pathnet/model/trans/ch1/pathnet.ckpt    --ovnet_ckpt=ovnet/model/trans/ch1/ovnet.ckpt    --data_dir=data/chinese1 --chinese1=True
# python vectorize.py --data_type=chinese --test_dir=test/ch2    --num_test_files=100 --image_height=48 --image_width=48 --pathnet_ckpt=pathnet/model/trans/ch2/pathnet.ckpt    --ovnet_ckpt=ovnet/model/trans/ch2/ovnet.ckpt    --data_dir=data/chinese2 --chinese1=False
# python vectorize.py --data_type=line    --test_dir=test/line   --num_test_files=100 --image_height=48 --image_width=48 --pathnet_ckpt=pathnet/model/no_trans/line/pathnet.ckpt   --ovnet_ckpt=ovnet/model/no_trans/line/ovnet.ckpt
# python vectorize.py --data_type=sketch  --test_dir=test/sketch --num_test_files=100 --image_height=64 --image_width=48 --pathnet_ckpt=pathnet/model/trans/sketch/pathnet.ckpt --ovnet_ckpt=ovnet/model/trans/sketch/ovnet.ckpt --data_dir=data/sketch --max_num_labels=256

# no trans-no trans: useless...
