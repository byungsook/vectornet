python ovnet_train.py --train_on=sketch --data_dir=../data/sketch --transform=True --log_dir=log/sketch --max_steps=50000

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
