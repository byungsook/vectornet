


# # 03-03-17 Fri., new training for r1.0 without transform -> 50k is enough
# python pathnet_train.py --train_on=chinese --log_dir=log/ch1    --max_steps=100000 --transform=False --data_dir=../data/chinese1 --chinese1=True
# python pathnet_train.py --train_on=chinese --log_dir=log/ch2    --max_steps=100000 --transform=False --data_dir=../data/chinese2 --chinese1=False
# python pathnet_train.py --train_on=line    --log_dir=log/line   --max_steps=100000
# python pathnet_train.py --train_on=sketch  --log_dir=log/sketch --max_steps=100000 --transform=False --data_dir=../data/sketch

# python pathnet_train.py --train_on=chinese --log_dir=log/trans/ch1    --max_steps=50000 --transform=True --data_dir=../data/chinese1 --chinese1=True
# python pathnet_train.py --train_on=chinese --log_dir=log/trans/ch2    --max_steps=100000 --transform=True --data_dir=../data/chinese2 --chinese1=False
# python pathnet_train.py --train_on=sketch  --log_dir=log/trans/sketch --max_steps=50000  --transform=True --data_dir=../data/sketch
