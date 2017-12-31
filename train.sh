# pathnet
python main.py --archi=path --tag=lin --dataset=line
python main.py --archi=path --tag=lin --dataset=ch
python main.py --archi=path --tag=lin --dataset=kanji
python main.py --archi=path --tag=lin --dataset=baseball --height=128 --width=128 --lr=0.002
python main.py --archi=path --tag=lin --dataset=cat      --height=128 --width=128 --lr=0.002
python main.py --archi=path --tag=lin --dataset=multi    --height=128 --width=128 --lr=0.002

# overlapnet
python main.py --archi=overlap --tag=lin --dataset=line
python main.py --archi=overlap --tag=lin --dataset=ch
python main.py --archi=overlap --tag=lin --dataset=kanji
python main.py --archi=overlap --tag=lin --dataset=baseball --height=128 --width=128 --lr=0.002
python main.py --archi=overlap --tag=lin --dataset=cat      --height=128 --width=128 --lr=0.002
python main.py --archi=overlap --tag=lin --dataset=multi    --height=128 --width=128 --lr=0.002