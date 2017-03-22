# # 21-03-17 Tue., change ilr from 0.01 to 0.005 (only ch2), loss without scale factor
python pathnet_train.py --train_on=chinese --log_dir=log/no_trans_64/ch2 --max_steps=50000 --transform=False --image_width=64 --image_height=64 --batch_size=8 --initial_learning_rate=0.005 --data_dir=../data/chinese2 --chinese1=False
python pathnet_train.py --train_on=chinese --log_dir=log/no_trans_64/ch1 --max_steps=50000 --transform=False --image_width=64 --image_height=64 --batch_size=8 --initial_learning_rate=0.010 --data_dir=../data/chinese1 --chinese1=True
python pathnet_train.py --train_on=line    --log_dir=log/no_trans_64/line --max_steps=50000 --image_width=64 --image_height=64 --batch_size=8 --initial_learning_rate=0.010 --max_stroke_width=2 --num_paths=4

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
