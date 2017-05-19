# python pathnet_train.py --train_on=line --log_dir=log/no_trans_128/line --max_steps=50000 --image_width=128 --image_height=128 --batch_size=16 --initial_learning_rate=0.005 --num_paths=4 --path_type=2 --max_stroke_width=5

# fidelity
# python pathnet_train.py --train_on=fidelity --log_dir=log/no_trans_128/fidelity --max_steps=50000 --image_width=128 --image_height=128 --batch_size=16 --initial_learning_rate=0.005 --data_dir=../data/fidelity
python pathnet_train.py --train_on=fidelity --log_dir=log/no_trans_128/fidelity_256_c64 --max_steps=50000 --original_size=256 --image_width=64 --image_height=64 --batch_size=16 --initial_learning_rate=0.005 --data_dir=../data/fidelity


# python pathnet_train.py --train_on=line --log_dir=log/no_trans_128/line --max_steps=50000 --transform=False --image_width=128 --image_height=96 --batch_size=16 --initial_learning_rate=0.005

# bicycle!
# python pathnet_train.py --train_on=sketch --log_dir=log/no_trans_128/bicycle_cont --checkpoint_dir=log/no_trans_128/bicycle --max_steps=50000 --transform=False --image_width=128 --image_height=96 --batch_size=8 --initial_learning_rate=0.005 --data_dir=../data/bicycle

# python pathnet_train.py --train_on=sketch --log_dir=log/no_trans_128/bicycle --max_steps=50000 --transform=False --image_width=128 --image_height=96 --batch_size=8 --initial_learning_rate=0.005 --data_dir=../data/bicycle
# python pathnet_train.py --train_on=sketch --log_dir=log/no_trans_256/bicycle --max_steps=50000 --transform=False --image_width=256 --image_height=256 --batch_size=8 --initial_learning_rate=0.005 --data_dir=../data/bicycle --min_prop=0.1

# python pathnet_train.py --train_on=sketch2 --log_dir=log/no_trans_256/sketch2 --max_steps=50000 --transform=False --image_width=256 --image_height=256 --batch_size=4 --initial_learning_rate=0.001 --data_dir=../data/sketch_schneider --stroke_width=1

# python pathnet_train.py --train_on=sketch2 --log_dir=log/no_trans_128/sketch2_l --max_steps=50000 --transform=False --image_width=128 --image_height=128 --batch_size=8 --initial_learning_rate=0.005 --data_dir=../data/sketch_schneider_l
# python pathnet_train.py --train_on=sketch2 --log_dir=log/no_trans_256/sketch2 --max_steps=50000 --transform=False --image_width=256 --image_height=256 --batch_size=8 --initial_learning_rate=0.005 --data_dir=../data/sketch_schneider

# # 21-03-17 Tue., change ilr from 0.01 to 0.005 (only ch2), loss without scale factor
# python pathnet_train.py --train_on=sketch2 --log_dir=log/no_trans_128/sketch2 --max_steps=50000 --transform=False --image_width=128 --image_height=128 --batch_size=8 --initial_learning_rate=0.005 --data_dir=../data/sketch_schneider


# # 21-03-17 Tue., change ilr from 0.01 to 0.005 (only ch2), loss without scale factor
# python pathnet_train.py --train_on=chinese --log_dir=log/no_trans_64/ch2 --max_steps=50000 --transform=False --image_width=64 --image_height=64 --batch_size=8 --initial_learning_rate=0.005 --data_dir=../data/chinese2 --chinese1=False
# python pathnet_train.py --train_on=chinese --log_dir=log/no_trans_64/ch1 --max_steps=50000 --transform=False --image_width=64 --image_height=64 --batch_size=8 --initial_learning_rate=0.010 --data_dir=../data/chinese1 --chinese1=True
# python pathnet_train.py --train_on=line    --log_dir=log/no_trans_64/line --max_steps=50000 --image_width=64 --image_height=64 --batch_size=8 --initial_learning_rate=0.010 --max_stroke_width=2 --num_paths=4

# # 20-03-17 Mon., new train on 64^2, 50000 steps, without transform
# python pathnet_train.py --train_on=chinese --log_dir=log/no_trans_64/ch1 --max_steps=50000 --transform=False --image_width=64 --image_height=64 --batch_size=8 --data_dir=../data/chinese1 --chinese1=True
# python pathnet_train.py --train_on=chinese --log_dir=log/no_trans_64/ch2 --max_steps=50000 --transform=False --image_width=64 --image_height=64 --batch_size=8 --data_dir=../data/chinese2 --chinese1=False

# # 03-03-17 Fri., new training for r1.0 without transform -> 50k is enough
# python pathnet_train.py --train_on=chinese --log_dir=log/ch1    --max_steps=100000 --transform=False --data_dir=../data/chinese1 --chinese1=True
# python pathnet_train.py --train_on=chinese --log_dir=log/ch2    --max_steps=100000 --transform=False --data_dir=../data/chinese2 --chinese1=False
# python pathnet_train.py --train_on=line    --log_dir=log/line   --max_steps=100000
# python pathnet_train.py --train_on=sketch  --log_dir=log/sketch --max_steps=100000 --transform=False --data_dir=../data/sketch

# python pathnet_train.py --train_on=chinese --log_dir=log/trans/ch1    --max_steps=50000 --transform=True --data_dir=../data/chinese1 --chinese1=True
# python pathnet_train.py --train_on=chinese --log_dir=log/trans/ch2    --max_steps=100000 --transform=True --data_dir=../data/chinese2 --chinese1=False
# python pathnet_train.py --train_on=sketch  --log_dir=log/trans/sketch --max_steps=50000  --transform=True --data_dir=../data/sketch
