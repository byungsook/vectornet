:::: 22-02-17 Thu., new train..
::python ovnet_train.py --train_on=chinese --chinese1=True  --data_dir=../data/chinese1 --transform=True --log_dir=log/ch1 --max_steps=50000 --decay_steps=30000 --initial_learning_rate=0.01 --batch_size=8 --image_width=128 --image_height=128
::python ovnet_train.py --train_on=chinese --chinese1=False --data_dir=../data/chinese2 --transform=True --log_dir=log/ch2 --max_steps=50000 --decay_steps=30000 --initial_learning_rate=0.01 --batch_size=8 --image_width=128 --image_height=128
::python ovnet_train.py --train_on=line --transform=True   --log_dir=log/line   --max_steps=50000 --decay_steps=30000 --initial_learning_rate=0.01 --batch_size=8 --image_width=128 --image_height=128
python ovnet_train.py --train_on=line --transform=True  --pretrained_model_checkpoint_path=log/line/ovnet.ckpt-45000 --log_dir=log/line_cont   --max_steps=50000 --decay_steps=30000 --initial_learning_rate=0.01 --batch_size=8 --image_width=128 --image_height=128
python ovnet_train.py --train_on=sketch --transform=True --log_dir=log/sketch --max_steps=50000 --decay_steps=30000 --initial_learning_rate=0.01 --batch_size=8 --image_width=128 --image_height=128
python ovnet_train.py --train_on=hand --transform=True   --log_dir=log/hand   --max_steps=50000 --decay_steps=30000 --initial_learning_rate=0.01 --batch_size=8 --image_width=128 --image_height=128

:::: 19-02-17 Sun., new train..
::python ovnet_train.py --train_on=line --log_dir=log/line --max_steps=50000 --transform=True --log_dir=log/line --max_steps=50000 --initial_learning_rate=0.01 --decay_steps=50000 --batch_size=8 --image_width=128 --image_height=128
::python ovnet_train.py --train_on=chinese --chinese1=True --data_dir=../data/chinese1 --transform=True --log_dir=log/ch1 --max_steps=50000 --initial_learning_rate=0.01 --decay_steps=50000 --batch_size=8 --image_width=128 --image_height=128
::python ovnet_train.py --train_on=chinese --chinese1=False --data_dir=../data/chinese2 --transform=True --log_dir=log/ch2 --max_steps=50000 --initial_learning_rate=0.01 --decay_steps=50000 --batch_size=8 --image_width=128 --image_height=128
::python ovnet_train.py --train_on=sketch --data_dir=../data/sketch --transform=True --log_dir=log/sketch --max_steps=50000 --initial_learning_rate=0.01 --decay_steps=50000 --batch_size=8 --image_width=128 --image_height=128
::python ovnet_train.py --train_on=hand --data_dir=../data/hand --transform=True --log_dir=log/hand --max_steps=50000 --initial_learning_rate=0.01 --decay_steps=50000 --batch_size=8 --image_width=128 --image_height=128
