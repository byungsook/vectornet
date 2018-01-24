REM pathnet
python main.py --archi=path --tag=win --dataset=line
python main.py --archi=path --tag=win --dataset=ch
python main.py --archi=path --tag=win --dataset=kanji
python main.py --archi=path --tag=win --dataset=baseball --height=128 --width=128 --lr=0.002
python main.py --archi=path --tag=win --dataset=cat      --height=128 --width=128 --lr=0.002

REM overlapnet
python main.py --archi=overlap --tag=win --dataset=line
python main.py --archi=overlap --tag=win --dataset=ch
python main.py --archi=overlap --tag=win --dataset=kanji
python main.py --archi=overlap --tag=win --dataset=baseball --height=128 --width=128 --lr=0.002
python main.py --archi=overlap --tag=win --dataset=cat      --height=128 --width=128 --lr=0.002