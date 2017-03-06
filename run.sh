# no trans-no trans: useless...

# trans-trans
python vectorize.py --data_type=chinese --test_dir=test/ch1    --num_test_files=8 --image_height=48 --image_width=48 --pathnet_ckpt=pathnet/model/trans/ch1/pathnet.ckpt    --ovnet_ckpt=ovnet/model/trans/ch1/ovnet.ckpt    --data_dir=data/chinese1 --chinese1=True
python vectorize.py --data_type=chinese --test_dir=test/ch2    --num_test_files=8 --image_height=64 --image_width=64 --pathnet_ckpt=pathnet/model/trans/ch2/pathnet.ckpt    --ovnet_ckpt=ovnet/model/trans/ch2/ovnet.ckpt    --data_dir=data/chinese2 --chinese1=False
python vectorize.py --data_type=line    --test_dir=test/line   --num_test_files=8 --image_height=64 --image_width=64 --pathnet_ckpt=pathnet/model/no_trans/line/pathnet.ckpt   --ovnet_ckpt=ovnet/model/no_trans/line/ovnet.ckpt
python vectorize.py --data_type=sketch  --test_dir=test/sketch --num_test_files=8 --image_height=64 --image_width=48 --pathnet_ckpt=pathnet/model/trans/sketch/pathnet.ckpt --ovnet_ckpt=ovnet/model/trans/sketch/ovnet.ckpt --data_dir=data/sketch --max_num_labels=256
