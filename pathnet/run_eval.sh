python pathnet_eval.py --train_on=line    --eval_dir=eval/64/line    --checkpoint_dir=model/no_trans_64/line --image_width=64 --image_height=64 --batch_size=8 --max_images=8 --num_epoch=1 --data_dir=../data/line_ov

# qdraw
# python pathnet_eval.py --train_on=qdraw --eval_dir=eval/qdraw/test/baseball_128  --checkpoint_dir=model/no_trans_128/qdraw_baseball_128 --image_width=128 --image_height=128 --batch_size=16 --max_images=16 --num_epoch=1 --data_dir=../data/qdraw_baseball_128 --file_list=test.txt
# python pathnet_eval.py --train_on=qdraw --eval_dir=eval/qdraw/test/cat_128       --checkpoint_dir=model/no_trans_128/qdraw_stitches_128 --image_width=128 --image_height=128 --batch_size=16 --max_images=16 --num_epoch=1 --data_dir=../data/qdraw_cat_128      --file_list=test.txt
# python pathnet_eval.py --train_on=qdraw --eval_dir=eval/qdraw/test/stitches_128  --checkpoint_dir=model/no_trans_128/qdraw_cat_128      --image_width=128 --image_height=128 --batch_size=16 --max_images=16 --num_epoch=1 --data_dir=../data/qdraw_stitches_128 --file_list=test.txt
# python pathnet_eval.py --train_on=qdraw --eval_dir=eval/qdraw/train/baseball_128 --checkpoint_dir=model/no_trans_128/qdraw_baseball_128 --image_width=128 --image_height=128 --batch_size=16 --max_images=16 --num_epoch=1 --data_dir=../data/qdraw_baseball_128 --file_list=train.txt
# python pathnet_eval.py --train_on=qdraw --eval_dir=eval/qdraw/train/cat_128      --checkpoint_dir=model/no_trans_128/qdraw_stitches_128 --image_width=128 --image_height=128 --batch_size=16 --max_images=16 --num_epoch=1 --data_dir=../data/qdraw_cat_128      --file_list=train.txt
# python pathnet_eval.py --train_on=qdraw --eval_dir=eval/qdraw/train/stitches_128 --checkpoint_dir=model/no_trans_128/qdraw_cat_128      --image_width=128 --image_height=128 --batch_size=16 --max_images=16 --num_epoch=1 --data_dir=../data/qdraw_stitches_128 --file_list=train.txt


# python pathnet_eval.py --train_on=fidelity --eval_dir=eval/no_trans_128/fidelity_train_256 --checkpoint_dir=model/no_trans_128/fidelity --image_width=64 --image_height=64 --batch_size=16 --max_images=16 --num_epoch=3 --data_dir=../data/fidelity --file_list=train.txt
# python pathnet_eval.py --train_on=fidelity --eval_dir=eval/no_trans_128/fidelity_test_256  --checkpoint_dir=model/no_trans_128/fidelity --image_width=64 --image_height=64 --batch_size=16 --max_images=16 --num_epoch=3 --data_dir=../data/fidelity --file_list=test.txt


# python pathnet_eval.py --train_on=fidelity --eval_dir=eval/no_trans_128/fidelity_train --checkpoint_dir=model/no_trans_128/fidelity --image_width=128 --image_height=128 --batch_size=16 --max_images=16 --num_epoch=3 --data_dir=../data/fidelity --file_list=train.txt
# python pathnet_eval.py --train_on=fidelity --eval_dir=eval/no_trans_128/fidelity_test  --checkpoint_dir=model/no_trans_128/fidelity --image_width=128 --image_height=128 --batch_size=16 --max_images=16 --num_epoch=3 --data_dir=../data/fidelity --file_list=test.txt

# python pathnet_eval.py --train_on=sketch --eval_dir=eval/no_trans_128/bicycle   --checkpoint_dir=model/no_trans_128/bicycle --transform=False --image_width=128 --image_height=96 --batch_size=8 --max_images=8 --num_epoch=3 --data_dir=../data/bicycle
# python pathnet_eval.py --train_on=sketch --eval_dir=eval/no_trans_128/car       --checkpoint_dir=model/no_trans_128/car     --transform=False --image_width=128 --image_height=96 --batch_size=8 --max_images=8 --num_epoch=3 --data_dir=../data/car
# python pathnet_eval.py --train_on=sketch --eval_dir=eval/no_trans_128/snail     --checkpoint_dir=model/no_trans_128/snail   --transform=False --image_width=128 --image_height=96 --batch_size=8 --max_images=8 --num_epoch=3 --data_dir=../data/snail

# python pathnet_eval.py --train_on=sketch2 --eval_dir=eval/no_trans_128/sketch2_l  --checkpoint_dir=log/no_trans_128/sketch2_l --transform=False --image_width=128 --image_height=128 --batch_size=8 --max_images=8 --num_epoch=3 --data_dir=../data/sketch_schneider_l
# python pathnet_eval.py --train_on=sketch2 --eval_dir=eval/no_trans_128/sketch2  --checkpoint_dir=log/no_trans_128/sketch2 --transform=False --image_width=128 --image_height=128 --batch_size=8 --max_images=8 --num_epoch=3 --data_dir=../data/sketch_schneider

# # 21-03-17 Tue., after training on 64^2, 50000 steps, without transform
# python pathnet_eval.py --train_on=chinese --eval_dir=eval/no_trans_64/ch1     --pretrained_model_checkpoint_path=log/no_trans_64/ch1/pathnet.ckpt-50000 --transform=False --image_width=64 --image_height=64 --batch_size=8 --max_images=8 --num_epoch=3 --data_dir=../data/chinese1 --chinese1=True
# python pathnet_eval.py --train_on=chinese --eval_dir=eval/no_trans_64/ch1_ch2 --pretrained_model_checkpoint_path=log/no_trans_64/ch1/pathnet.ckpt-50000 --transform=False --image_width=64 --image_height=64 --batch_size=8 --max_images=8 --num_epoch=3 --data_dir=../data/chinese2 --chinese1=False
# python pathnet_eval.py --train_on=chinese --eval_dir=eval/no_trans_64/ch2     --pretrained_model_checkpoint_path=log/no_trans_64/ch2/pathnet.ckpt-50000 --transform=False --image_width=64 --image_height=64 --batch_size=8 --max_images=8 --num_epoch=3 --data_dir=../data/chinese2 --chinese1=False
# python pathnet_eval.py --train_on=chinese --eval_dir=eval/no_trans_64/ch2_ch1 --pretrained_model_checkpoint_path=log/no_trans_64/ch2/pathnet.ckpt-50000 --transform=False --image_width=64 --image_height=64 --batch_size=8 --max_images=8 --num_epoch=3 --data_dir=../data/chinese1 --chinese1=True
# python pathnet_eval.py --train_on=line    --eval_dir=eval/no_trans_64/line    --pretrained_model_checkpoint_path=log/no_trans_64/line/pathnet.ckpt-50000 --image_width=64 --image_height=64 --batch_size=8 --max_images=8 --num_epoch=3 --max_stroke_width=2 --num_paths=4
