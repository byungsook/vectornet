# pathnet
python main.py --archi=path --log_step=100 --batch_size=8  --num_worker=8  --lr=0.005 --lr_update_step=20000 --max_step=50000 --dataset=line
python main.py --archi=path --log_step=100 --batch_size=8  --num_worker=8  --lr=0.005 --lr_update_step=20000 --max_step=50000 --dataset=ch
python main.py --archi=path --log_step=100 --batch_size=8  --num_worker=8  --lr=0.005 --lr_update_step=20000 --max_step=50000 --dataset=kanji
python main.py --archi=path --log_step=100 --batch_size=16 --num_worker=16 --lr=0.005 --lr_update_step=20000 --max_step=50000 --dataset=baseball --height=128 --width=128
python main.py --archi=path --log_step=100 --batch_size=16 --num_worker=16 --lr=0.005 --lr_update_step=20000 --max_step=50000 --dataset=cat      --height=128 --width=128
python main.py --archi=path --log_step=100 --batch_size=16 --num_worker=16 --lr=0.005 --lr_update_step=20000 --max_step=50000 --dataset=multi    --height=128 --width=128

# overlapnet
python main.py --archi=overlap --log_step=100 --batch_size=8  --num_worker=8  --lr=0.005 --lr_update_step=20000 --max_step=50000 --dataset=line
python main.py --archi=overlap --log_step=100 --batch_size=8  --num_worker=8  --lr=0.005 --lr_update_step=20000 --max_step=50000 --dataset=ch
python main.py --archi=overlap --log_step=100 --batch_size=8  --num_worker=8  --lr=0.005 --lr_update_step=20000 --max_step=50000 --dataset=kanji
python main.py --archi=overlap --log_step=100 --batch_size=16 --num_worker=16 --lr=0.005 --lr_update_step=20000 --max_step=50000 --dataset=baseball --height=128 --width=128
python main.py --archi=overlap --log_step=100 --batch_size=16 --num_worker=16 --lr=0.005 --lr_update_step=20000 --max_step=50000 --dataset=cat      --height=128 --width=128
python main.py --archi=overlap --log_step=100 --batch_size=16 --num_worker=16 --lr=0.005 --lr_update_step=20000 --max_step=50000 --dataset=multi    --height=128 --width=128

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