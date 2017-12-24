# pathnet
# python main.py --archi=path --tag=01 --log_step=100 --batch_size=8  --num_worker=8  --lr=0.005 --lr_update_step=20000 --max_step=50000 --data_dir=/media/kimby/Data/Polybox/dev/vectornet2/data --log_dir=/media/kimby/Data/Polybox/dev/vectornet2/log --dataset=line
# python main.py --archi=path --tag=01 --log_step=100 --batch_size=8  --num_worker=8  --lr=0.005 --lr_update_step=20000 --max_step=50000 --data_dir=/media/kimby/Data/Polybox/dev/vectornet2/data --log_dir=/media/kimby/Data/Polybox/dev/vectornet2/log --dataset=ch
# python main.py --archi=path --tag=01 --log_step=100 --batch_size=8  --num_worker=8  --lr=0.005 --lr_update_step=20000 --max_step=50000 --data_dir=/media/kimby/Data/Polybox/dev/vectornet2/data --log_dir=/media/kimby/Data/Polybox/dev/vectornet2/log --dataset=kanji
python main.py --archi=path --tag=01 --log_step=100 --batch_size=8 --num_worker=8 --lr=0.002 --lr_update_step=20000 --max_step=50000 --data_dir=/media/kimby/Data/Polybox/dev/vectornet2/data --log_dir=/media/kimby/Data/Polybox/dev/vectornet2/log --dataset=baseball --height=128 --width=128
python main.py --archi=path --tag=01 --log_step=100 --batch_size=8 --num_worker=8 --lr=0.002 --lr_update_step=20000 --max_step=50000 --data_dir=/media/kimby/Data/Polybox/dev/vectornet2/data --log_dir=/media/kimby/Data/Polybox/dev/vectornet2/log --dataset=cat      --height=128 --width=128
python main.py --archi=path --tag=01 --log_step=100 --batch_size=8 --num_worker=8 --lr=0.002 --lr_update_step=20000 --max_step=50000 --data_dir=/media/kimby/Data/Polybox/dev/vectornet2/data --log_dir=/media/kimby/Data/Polybox/dev/vectornet2/log --dataset=multi    --height=128 --width=128
python main.py --archi=path --tag=sw2 --log_step=100 --batch_size=8  --num_worker=8  --lr=0.005 --lr_update_step=20000 --max_step=50000 --data_dir=/media/kimby/Data/Polybox/dev/vectornet2/data --log_dir=/media/kimby/Data/Polybox/dev/vectornet2/log --dataset=line

# overlapnet
# python main.py --archi=overlap --tag=01 --log_step=100 --batch_size=8  --num_worker=8  --lr=0.005 --lr_update_step=20000 --max_step=50000 --data_dir=/media/kimby/Data/Polybox/dev/vectornet2/data --log_dir=/media/kimby/Data/Polybox/dev/vectornet2/log --dataset=line
python main.py --archi=overlap --tag=01 --log_step=100 --batch_size=8  --num_worker=8  --lr=0.005 --lr_update_step=20000 --max_step=50000 --data_dir=/media/kimby/Data/Polybox/dev/vectornet2/data --log_dir=/media/kimby/Data/Polybox/dev/vectornet2/log --dataset=ch
python main.py --archi=overlap --tag=01 --log_step=100 --batch_size=8  --num_worker=8  --lr=0.005 --lr_update_step=20000 --max_step=50000 --data_dir=/media/kimby/Data/Polybox/dev/vectornet2/data --log_dir=/media/kimby/Data/Polybox/dev/vectornet2/log --dataset=kanji
python main.py --archi=overlap --tag=01 --log_step=100 --batch_size=8 --num_worker=8 --lr=0.002 --lr_update_step=20000 --max_step=50000 --data_dir=/media/kimby/Data/Polybox/dev/vectornet2/data --log_dir=/media/kimby/Data/Polybox/dev/vectornet2/log --dataset=baseball --height=128 --width=128
python main.py --archi=overlap --tag=01 --log_step=100 --batch_size=8 --num_worker=8 --lr=0.002 --lr_update_step=20000 --max_step=50000 --data_dir=/media/kimby/Data/Polybox/dev/vectornet2/data --log_dir=/media/kimby/Data/Polybox/dev/vectornet2/log --dataset=cat      --height=128 --width=128
python main.py --archi=overlap --tag=01 --log_step=100 --batch_size=8 --num_worker=8 --lr=0.002 --lr_update_step=20000 --max_step=50000 --data_dir=/media/kimby/Data/Polybox/dev/vectornet2/data --log_dir=/media/kimby/Data/Polybox/dev/vectornet2/log --dataset=multi    --height=128 --width=128
python main.py --archi=overlap --tag=sw2 --log_step=100 --batch_size=8  --num_worker=8  --lr=0.005 --lr_update_step=20000 --max_step=50000 --data_dir=/media/kimby/Data/Polybox/dev/vectornet2/data --log_dir=/media/kimby/Data/Polybox/dev/vectornet2/log --dataset=line

# test
# pathnet
# python main.py --archi=path --tag=test --log_step=10 --batch_size=8 --num_worker=8 --lr=0.005 --lr_update_step=20000 --max_step=500 --dataset=line
# python main.py --archi=path --tag=test --log_step=10 --batch_size=8 --num_worker=8 --lr=0.005 --lr_update_step=20000 --max_step=500 --dataset=ch
# python main.py --archi=path --tag=test --log_step=10 --batch_size=8 --num_worker=8 --lr=0.005 --lr_update_step=20000 --max_step=500 --dataset=kanji
# python main.py --archi=path --tag=test --log_step=10 --batch_size=8 --num_worker=8 --lr=0.005 --lr_update_step=20000 --max_step=500 --dataset=baseball --height=128 --width=128
# python main.py --archi=path --tag=test --log_step=10 --batch_size=8 --num_worker=8 --lr=0.005 --lr_update_step=20000 --max_step=500 --dataset=cat --height=128 --width=128
# python main.py --archi=path --tag=test --log_step=10 --batch_size=8 --num_worker=8 --lr=0.005 --lr_update_step=20000 --max_step=500 --dataset=multi --height=128 --width=128

# overlapnet
# python main.py --archi=overlap --tag=test --log_step=10 --batch_size=8 --num_worker=8 --lr=0.005 --lr_update_step=20000 --max_step=500 --dataset=line
# python main.py --archi=overlap --tag=test --log_step=10 --batch_size=8 --num_worker=8 --lr=0.005 --lr_update_step=20000 --max_step=500 --dataset=ch
# python main.py --archi=overlap --tag=test --log_step=10 --batch_size=8 --num_worker=8 --lr=0.005 --lr_update_step=20000 --max_step=500 --dataset=kanji
# python main.py --archi=overlap --tag=test --log_step=10 --batch_size=8 --num_worker=8 --lr=0.005 --lr_update_step=20000 --max_step=500 --dataset=baseball --height=128 --width=128
# python main.py --archi=overlap --tag=test --log_step=10 --batch_size=8 --num_worker=8 --lr=0.005 --lr_update_step=20000 --max_step=500 --dataset=cat --height=128 --width=128
# python main.py --archi=overlap --tag=test --log_step=10 --batch_size=8 --num_worker=8 --lr=0.005 --lr_update_step=20000 --max_step=500 --dataset=multi --height=128 --width=128