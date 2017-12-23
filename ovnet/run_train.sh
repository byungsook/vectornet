python ovnet_train.py --train_on=line    --log_dir=log/64/line --max_steps=50000 --transform=False --image_width=64 --image_height=64 --batch_size=8 --use_iou=False --path_type=2 --num_paths=4 --max_stroke_width=4 --initial_learning_rate=0.005 

# python ovnet_train_sketch.py --train_on=qdraw --log_dir=log/l2_128/qdraw/chandelier --max_steps=10 --image_width=128 --image_height=128 --batch_size=16 --initial_learning_rate=0.005 --data_dir=../data/qdraw_chandelier_128 --use_iou=False
# python ovnet_train_sketch.py --train_on=qdraw --log_dir=log/l2_128/qdraw/elephant   --max_steps=10 --image_width=128 --image_height=128 --batch_size=16 --initial_learning_rate=0.005 --data_dir=../data/qdraw_elephant_128   --use_iou=False
# python ovnet_train_sketch.py --train_on=qdraw --log_dir=log/l2_128/qdraw/mix        --max_steps=50000 --image_width=128 --image_height=128 --batch_size=16 --initial_learning_rate=0.005 --data_dir=../data/qdraw_mix_128        --use_iou=False

# qdraw
# python ovnet_train_sketch.py --train_on=qdraw --log_dir=log/l2_128/qdraw_baseball --max_steps=50000 --image_width=128 --image_height=128 --batch_size=16 --initial_learning_rate=0.005 --data_dir=../data/qdraw_baseball_128 --use_iou=False
# python ovnet_train_sketch.py --train_on=qdraw --log_dir=log/l2_128/qdraw_stitches --max_steps=50000 --image_width=128 --image_height=128 --batch_size=16 --initial_learning_rate=0.005 --data_dir=../data/qdraw_stitches_128 --use_iou=False
# python ovnet_train_sketch.py --train_on=qdraw --log_dir=log/l2_128/qdraw_cat      --max_steps=50000 --image_width=128 --image_height=128 --batch_size=16 --initial_learning_rate=0.005 --data_dir=../data/qdraw_cat_128      --use_iou=False

# python ovnet_train_sketch.py --train_on=fidelity --log_dir=log/l2_64/fidelity_256_c64  --max_steps=50000 --transform=False --original_size=256  --image_width=64 --image_height=64 --batch_size=8 --use_iou=False --data_dir=../data/fidelity
# python ovnet_train_sketch.py --train_on=fidelity --log_dir=log/l2_64/fidelity_1024_c64 --max_steps=50000 --transform=False --original_size=1024 --image_width=64 --image_height=64 --batch_size=8 --use_iou=False --data_dir=../data/fidelity


# python ovnet_train.py --train_on=line    --log_dir=log/iou_64/line --max_steps=50000 --transform=False --image_width=64 --image_height=64 --batch_size=8 --use_iou=True --path_type=0 --num_paths=4 --max_stroke_width=5
# ./run_eval.sh
# ../run_png.sh
# python ovnet_train.py --train_on=chinese --log_dir=log/l2_64/ch1  --max_steps=50000 --transform=False --image_width=64 --image_height=64 --batch_size=8 --use_iou=False --data_dir=../data/chinese1 --chinese1=True
# python ovnet_train.py --train_on=chinese --log_dir=log/l2_64/ch2  --max_steps=50000 --transform=False --image_width=64 --image_height=64 --batch_size=8 --use_iou=False --data_dir=../data/chinese2 --chinese1=False
# python ovnet_train.py --train_on=line    --log_dir=log/l2_64/line --max_steps=50000 --transform=False --image_width=64 --image_height=64 --batch_size=8 --use_iou=False


# python ovnet_train.py --train_on=sketch2 --log_dir=log/no_trans_128/sketch2_l --max_steps=50000 --transform=False --image_width=128 --image_height=128 --batch_size=8 --data_dir=../data/sketch_schneider_l
# ./run_eval.sh

# # sketch and others?
# python ovnet_train.py --train_on=sketch2 --log_dir=log/no_trans_128/sketch2 --max_steps=50000 --transform=False --image_width=128 --image_height=128 --batch_size=8 --data_dir=../data/sketch_schneider
# ./run_eval.sh

# # 20-03-17 Mon. train on IoU metric, 64^2, 50000 steps, without transform
# python ovnet_train.py --train_on=chinese --log_dir=log/no_trans_64/ch1 --max_steps=50000 --transform=False --image_width=64 --image_height=64 --batch_size=8 --data_dir=../data/chinese1 --chinese1=True
# python ovnet_train.py --train_on=chinese --log_dir=log/no_trans_64/ch2 --max_steps=50000 --transform=False --image_width=64 --image_height=64 --batch_size=8 --data_dir=../data/chinese2 --chinese1=False
# python ovnet_train.py --train_on=line --log_dir=log/no_trans_64/line --max_steps=50000 --image_width=64 --image_height=64 --batch_size=8 --max_stroke_width=2 --num_paths=4

# python ovnet_train.py --train_on=sketch --data_dir=../data/sketch --transform=True --log_dir=log/sketch --max_steps=50000

# 28-02-17 Tue., without transform
# python ovnet_train.py --train_on=line --log_dir=log/line --max_steps=50000
# python ovnet_train.py --train_on=chinese --chinese1=True  --data_dir=../data/chinese1 --transform=False --log_dir=log/ch1 --max_steps=50000
# python ovnet_train.py --train_on=chinese --chinese1=False --data_dir=../data/chinese2 --transform=False --log_dir=log/ch2 --max_steps=50000
# python ovnet_train.py --train_on=sketch --data_dir=../data/sketch --transform=False --log_dir=log/sketch --max_steps=50000
# python ovnet_train.py --train_on=hand --data_dir=../data/hand --transform=False --log_dir=log/hand --max_steps=50000

# 26-01-17 thu, train 20k->50k, forgot to set batch_size 8...
# python ovnet_train.py --train_on=chinese --chinese1=True --data_dir=../data/chinese1 --transform=True --log_dir=log/ch1 --max_steps=50000 --pretrained_model_checkpoint_path=log/old/ch1/ovnet.ckpt
# python ovnet_train.py --train_on=chinese --chinese1=False --data_dir=../data/chinese2 --transform=True --log_dir=log/ch2 --max_steps=50000 --pretrained_model_checkpoint_path=log/old/ch2/ovnet.ckpt
# python ovnet_train.py --train_on=line --log_dir=log/line --max_steps=50000 --pretrained_model_checkpoint_path=log/old/line/ovnet.ckpt
# python ovnet_train.py --train_on=sketch --data_dir=../data/sketch --transform=True --log_dir=log/sketch --max_steps=50000 --batch_size=8
# python ovnet_train.py --train_on=hand --data_dir=../data/hand --transform=True --log_dir=log/hand --max_steps=50000

# 25-01-17 wed, re-train
# python ovnet_train.py --train_on=line --log_dir=log/line --max_steps=20000
# python ovnet_train.py --train_on=chinese --chinese1=True --data_dir=../data/chinese1 --transform=True --log_dir=log/ch1 --max_steps=20000
# python ovnet_train.py --train_on=chinese --chinese1=False --data_dir=../data/chinese2 --transform=True --log_dir=log/ch2 --max_steps=20000
# python ovnet_train.py --train_on=sketch --data_dir=../data/sketch --transform=True --log_dir=log/sketch --max_steps=20000
# python ovnet_train.py --train_on=hand --data_dir=../data/hand --transform=True --log_dir=log/hand --max_steps=20000
