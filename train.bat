REM pathnet
REM python main.py --archi=path    --tag=test --log_step=100 --batch_size=8 --num_worker=8 --lr=0.005 --lr_update_step=20000 --max_step=50000 --dataset=line
REM python main.py --archi=path    --tag=test --log_step=10 --batch_size=8 --num_worker=8 --lr=0.005 --lr_update_step=20000 --max_step=1000 --dataset=ch
REM python main.py --archi=path    --tag=test --log_step=10 --batch_size=8 --num_worker=8 --lr=0.005 --lr_update_step=20000 --max_step=1000 --dataset=kanji
python main.py --archi=path    --tag=test --log_step=10 --batch_size=8 --num_worker=8 --lr=0.005 --lr_update_step=20000 --max_step=500 --dataset=baseball --height=128 --width=128
python main.py --archi=path    --tag=test --log_step=10 --batch_size=8 --num_worker=8 --lr=0.005 --lr_update_step=20000 --max_step=500 --dataset=cat --height=128 --width=128
python main.py --archi=path    --tag=test --log_step=10 --batch_size=8 --num_worker=8 --lr=0.005 --lr_update_step=20000 --max_step=500 --dataset=multi --height=128 --width=128

REM overlapnet
REM python main.py --archi=overlap --tag=test --log_step=100 --batch_size=8 --num_worker=8 --lr=0.005 --lr_update_step=20000 --max_step=50000 --dataset=line
REM python main.py --archi=overlap    --tag=test --log_step=10 --batch_size=8 --num_worker=8 --lr=0.005 --lr_update_step=20000 --max_step=1000 --dataset=ch
REM python main.py --archi=overlap    --tag=test --log_step=10 --batch_size=8 --num_worker=8 --lr=0.005 --lr_update_step=20000 --max_step=1000 --dataset=kanji
python main.py --archi=overlap    --tag=test --log_step=10 --batch_size=8 --num_worker=8 --lr=0.005 --lr_update_step=20000 --max_step=500 --dataset=baseball --height=128 --width=128
python main.py --archi=overlap    --tag=test --log_step=10 --batch_size=8 --num_worker=8 --lr=0.005 --lr_update_step=20000 --max_step=500 --dataset=cat --height=128 --width=128
python main.py --archi=overlap    --tag=test --log_step=10 --batch_size=8 --num_worker=8 --lr=0.005 --lr_update_step=20000 --max_step=500 --dataset=multi --height=128 --width=128


