

python main.py --archi=path --tag=l1 --use_l2=True  --log_step=100 --max_step=50000 --batch_size=8 --num_worker=4 --lr=0.005 --lr_update_step=20000 --dataset=line --data_dir='/home/kimby/polybox/dev/vectornet2/data'
python main.py --archi=path --tag=l2 --use_l2=False --log_step=100 --max_step=50000 --batch_size=8 --num_worker=4 --lr=0.005 --lr_update_step=20000 --dataset=line --data_dir='/home/kimby/polybox/dev/vectornet2/data'