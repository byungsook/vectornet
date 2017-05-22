# python ovnet_eval_sketch.py --train_on=qdraw --eval_dir=eval/l2_128/qdraw/baseball_test   --data_dir=../data/qdraw_baseball_128 --checkpoint_dir=model/l2_128/qdraw_baseball --batch_size=16 --max_images=16 --num_epoch=1 --threshold=0.5 --image_width=128 --image_height=128 --file_list=test.txt
python ovnet_eval_sketch.py --train_on=qdraw --eval_dir=eval/l2_128/qdraw/baseball_train  --data_dir=../data/qdraw_baseball_128 --checkpoint_dir=model/l2_128/qdraw_baseball --batch_size=16 --max_images=16 --num_epoch=1 --threshold=0.5 --image_width=128 --image_height=128 --file_list=train.txt
python ovnet_eval_sketch.py --train_on=qdraw --eval_dir=eval/l2_128/qdraw/cat_test        --data_dir=../data/qdraw_cat_128      --checkpoint_dir=model/l2_128/qdraw_cat      --batch_size=16 --max_images=16 --num_epoch=1 --threshold=0.5 --image_width=128 --image_height=128 --file_list=test.txt
python ovnet_eval_sketch.py --train_on=qdraw --eval_dir=eval/l2_128/qdraw/cat_train       --data_dir=../data/qdraw_cat_128      --checkpoint_dir=model/l2_128/qdraw_cat      --batch_size=16 --max_images=16 --num_epoch=1 --threshold=0.5 --image_width=128 --image_height=128 --file_list=train.txt
python ovnet_eval_sketch.py --train_on=qdraw --eval_dir=eval/l2_128/qdraw/stitches_test   --data_dir=../data/qdraw_stitches_128 --checkpoint_dir=model/l2_128/qdraw_stitches --batch_size=16 --max_images=16 --num_epoch=1 --threshold=0.5 --image_width=128 --image_height=128 --file_list=test.txt
python ovnet_eval_sketch.py --train_on=qdraw --eval_dir=eval/l2_128/qdraw/stitches_train  --data_dir=../data/qdraw_stitches_128 --checkpoint_dir=model/l2_128/qdraw_stitches --batch_size=16 --max_images=16 --num_epoch=1 --threshold=0.5 --image_width=128 --image_height=128 --file_list=train.txt


# l2
# python ovnet_eval.py --train_on=chinese --chinese1=True   --eval_dir=eval/l2_64/ch1_test   --data_dir=../data/chinese1 --pretrained_model_checkpoint_path=model/l2_64/ch1/ovnet.ckpt-50000 --batch_size=8 --max_images=8 --num_epoch=1 --transform=False  --threshold=0.5 --image_width=64 --image_height=64 --file_list=test.txt
# python ovnet_eval.py --train_on=chinese --chinese1=True   --eval_dir=eval/l2_64/ch1_train  --data_dir=../data/chinese1 --pretrained_model_checkpoint_path=model/l2_64/ch1/ovnet.ckpt-50000 --batch_size=8 --max_images=8 --num_epoch=1 --transform=False  --threshold=0.5 --image_width=64 --image_height=64 --file_list=train.txt
# python ovnet_eval.py --train_on=chinese --chinese1=False  --eval_dir=eval/l2_64/ch2_test   --data_dir=../data/chinese2 --pretrained_model_checkpoint_path=model/l2_64/ch2/ovnet.ckpt-50000 --batch_size=8 --max_images=8 --num_epoch=1 --transform=False  --threshold=0.5 --image_width=64 --image_height=64 --file_list=test.txt
# python ovnet_eval.py --train_on=chinese --chinese1=False  --eval_dir=eval/l2_64/ch2_train  --data_dir=../data/chinese2 --pretrained_model_checkpoint_path=model/l2_64/ch2/ovnet.ckpt-50000 --batch_size=8 --max_images=8 --num_epoch=1 --transform=False  --threshold=0.5 --image_width=64 --image_height=64 --file_list=train.txt
# python ovnet_eval.py --train_on=line --eval_dir=eval/l2_64/line --data_dir=../data/line_ov --pretrained_model_checkpoint_path=model/l2_64/line/ovnet.ckpt-50000 --batch_size=8 --max_images=8 --num_epoch=1 --threshold=0.5 --image_width=64 --image_height=64
# python ovnet_eval_sketch.py --train_on=fidelity --eval_dir=eval/l2_64/fidelity_test  --data_dir=../data/fidelity --checkpoint_dir=model/l2_64/fidelity --batch_size=8 --max_images=8 --num_epoch=1 --threshold=0.5 --image_width=64 --image_height=64 --file_list=test.txt  --num_epoch=50
# python ovnet_eval_sketch.py --train_on=fidelity --eval_dir=eval/l2_64/fidelity_train --data_dir=../data/fidelity --checkpoint_dir=model/l2_64/fidelity --batch_size=8 --max_images=8 --num_epoch=1 --threshold=0.5 --image_width=64 --image_height=64 --file_list=train.txt --num_epoch=10

# iou
# python ovnet_eval.py --train_on=chinese --chinese1=True   --eval_dir=eval/iou_64/ch1_test  --data_dir=../data/chinese1 --pretrained_model_checkpoint_path=model/no_trans_64/ch1/ovnet.ckpt-50000 --batch_size=8 --max_images=8 --num_epoch=1 --transform=False  --threshold=0.5 --image_width=64 --image_height=64 --file_list=test.txt
# python ovnet_eval.py --train_on=chinese --chinese1=False  --eval_dir=eval/iou_64/ch2_test  --data_dir=../data/chinese2 --pretrained_model_checkpoint_path=model/no_trans_64/ch2/ovnet.ckpt-50000 --batch_size=8 --max_images=8 --num_epoch=1 --transform=False  --threshold=0.5 --image_width=64 --image_height=64 --file_list=test.txt
# python ovnet_eval.py --train_on=chinese --chinese1=True   --eval_dir=eval/iou_64/ch1_train  --data_dir=../data/chinese1 --pretrained_model_checkpoint_path=model/no_trans_64/ch1/ovnet.ckpt-50000 --batch_size=8 --max_images=8 --num_epoch=1 --transform=False  --threshold=0.5 --image_width=64 --image_height=64 --file_list=train.txt
# python ovnet_eval.py --train_on=chinese --chinese1=False  --eval_dir=eval/iou_64/ch2_train  --data_dir=../data/chinese2 --pretrained_model_checkpoint_path=model/no_trans_64/ch2/ovnet.ckpt-50000 --batch_size=8 --max_images=8 --num_epoch=1 --transform=False  --threshold=0.5 --image_width=64 --image_height=64 --file_list=train.txt
# python ovnet_eval.py --train_on=line --eval_dir=eval/iou_64/line --data_dir=../data/line_ov --pretrained_model_checkpoint_path=model/no_trans_64/line/ovnet.ckpt-50000 --batch_size=8 --max_images=8 --num_epoch=1 --threshold=0.5 --image_width=64 --image_height=64



# python ovnet_eval.py --train_on=chinese --chinese1=True   --eval_dir=eval/iou_64/ch1   --data_dir=../data/chinese1 --pretrained_model_checkpoint_path=model/no_trans_64/ch1/ovnet.ckpt-50000 --batch_size=8 --max_images=8 --num_epoch=1 --transform=False  --threshold=0.5 --image_width=64 --image_height=64
# python ovnet_eval.py --train_on=chinese --chinese1=False  --eval_dir=eval/iou_64/ch2   --data_dir=../data/chinese2 --pretrained_model_checkpoint_path=model/no_trans_64/ch2/ovnet.ckpt-50000 --batch_size=8 --max_images=8 --num_epoch=1 --transform=False  --threshold=0.5 --image_width=64 --image_height=64
# python ovnet_eval.py --train_on=line --eval_dir=eval/iou_64/line --pretrained_model_checkpoint_path=model/no_trans_64/line/ovnet.ckpt-50000 --batch_size=8 --max_images=8 --num_epoch=1 --threshold=0.5 --image_width=64 --image_height=64


# python ovnet_eval.py --train_on=line --eval_dir=eval/iou_64/line  --pretrained_model_checkpoint_path=model/no_trans_64/line2/ovnet.ckpt-50000 --batch_size=8 --max_images=8 --num_epoch=1 --transform=False --image_width=64 --image_height=64  --path_type=0 --num_paths=4 --max_stroke_width=5

# # new
# python ovnet_eval.py --train_on=chinese --chinese1=True   --eval_dir=eval/iou_64/ch1   --data_dir=../data/chinese1 --pretrained_model_checkpoint_path=model/no_trans_64/ch1/ovnet.ckpt-50000 --batch_size=8 --max_images=8 --num_epoch=1 --transform=False  --threshold=0.5 --image_width=64 --image_height=64

# # old
# python ovnet_eval_old.py --train_on=chinese --chinese1=True   --eval_dir=eval/l2_64/ch1   --data_dir=../data/chinese1 --pretrained_model_checkpoint_path=model/l2_64/ch1/ovnet.ckpt --batch_size=8 --max_images=8 --num_epoch=1 --transform=False  --threshold=0.5 --image_width=64 --image_height=64
# python ovnet_eval_old.py --train_on=chinese --chinese1=False  --eval_dir=eval/l2_64/ch2   --data_dir=../data/chinese2 --pretrained_model_checkpoint_path=model/l2_64/ch2/ovnet.ckpt --batch_size=8 --max_images=8 --num_epoch=1 --transform=False  --threshold=0.5 --image_width=64 --image_height=64
# python ovnet_eval_old.py --train_on=line --eval_dir=eval/l2_64/line --pretrained_model_checkpoint_path=model/l2_64/line/ovnet.ckpt --batch_size=8 --max_images=8 --num_epoch=1 --threshold=0.5 --image_width=64 --image_height=64

# python ovnet_eval.py --train_on=sketch --eval_dir=eval/no_trans_128/bicycle   --checkpoint_dir=model/no_trans_128/bicycle --transform=False --image_width=128 --image_height=96 --batch_size=8 --max_images=8 --num_epoch=1 --data_dir=../data/bicycle
# python ovnet_eval.py --train_on=sketch --eval_dir=eval/no_trans_128/car       --checkpoint_dir=model/no_trans_128/car     --transform=False --image_width=128 --image_height=96 --batch_size=8 --max_images=8 --num_epoch=1 --data_dir=../data/car
# python ovnet_eval.py --train_on=sketch --eval_dir=eval/no_trans_128/snail     --checkpoint_dir=model/no_trans_128/snail   --transform=False --image_width=128 --image_height=96 --batch_size=8 --max_images=8 --num_epoch=1 --data_dir=../data/snail

# python ovnet_eval.py --train_on=sketch2 --eval_dir=eval/no_trans_128/sketch2_l --data_dir=../data/sketch_schneider_l --checkpoint_dir=log/no_trans_128/sketch2_l --image_width=128 --image_height=128 --batch_size=8 --max_images=8 --num_epoch=1 --transform=False
# python ovnet_eval.py --train_on=sketch2 --eval_dir=eval/no_trans_128/sketch2 --data_dir=../data/sketch_schneider --checkpoint_dir=log/no_trans_128/sketch2 --image_width=128 --image_height=128 --batch_size=8 --max_images=8 --num_epoch=1 --transform=False

# # 21-03-17 Tue. After training on IoU metric
# python ovnet_eval.py --train_on=chinese --eval_dir=eval/no_trans_64/ch1 --data_dir=../data/chinese1 --pretrained_model_checkpoint_path=log/no_trans_64/ch1/ovnet.ckpt-50000 --image_width=64 --image_height=64 --batch_size=8 --max_images=8 --num_epoch=1 --transform=False --chinese1=True
# python ovnet_eval.py --train_on=chinese --eval_dir=eval/no_trans_64/ch1_ch2 --data_dir=../data/chinese2 --pretrained_model_checkpoint_path=log/no_trans_64/ch1/ovnet.ckpt-50000 --image_width=64 --image_height=64 --batch_size=8 --max_images=8 --num_epoch=1 --transform=False --chinese1=False

# python ovnet_eval.py --train_on=chinese --eval_dir=eval/no_trans_64/ch2 --data_dir=../data/chinese2 --pretrained_model_checkpoint_path=log/no_trans_64/ch2/ovnet.ckpt-50000 --image_width=64 --image_height=64 --batch_size=8 --max_images=8 --num_epoch=1 --transform=False --chinese1=False
# python ovnet_eval.py --train_on=chinese --eval_dir=eval/no_trans_64/ch2_ch1 --data_dir=../data/chinese1 --pretrained_model_checkpoint_path=log/no_trans_64/ch2/ovnet.ckpt-50000 --image_width=64 --image_height=64 --batch_size=8 --max_images=8 --num_epoch=1 --transform=False --chinese1=True

# python ovnet_eval.py --train_on=line --eval_dir=eval/no_trans_64/line --pretrained_model_checkpoint_path=log/no_trans_64/line/ovnet.ckpt-50000 --image_width=64 --image_height=64 --batch_size=8 --max_images=8 --num_epoch=1 --max_stroke_width=2 --num_paths=4


# 02-03-17 Thu., IoU cross eval
#  ch1: trained on no trans
# python ovnet_eval.py --train_on=chinese --chinese1=True   --eval_dir=eval/no_trans/ch1/ch1_n --data_dir=../data/chinese1 --pretrained_model_checkpoint_path=model/no_trans/ch1/ovnet.ckpt --batch_size=8 --max_images=8 --num_epoch=1 --transform=False --threshold=0.5
# python ovnet_eval.py --train_on=chinese --chinese1=True   --eval_dir=eval/no_trans/ch1/ch1   --data_dir=../data/chinese1 --pretrained_model_checkpoint_path=model/no_trans/ch1/ovnet.ckpt --batch_size=8 --max_images=8 --num_epoch=3 --transform=True  --threshold=0.5
# python ovnet_eval.py --train_on=chinese --chinese1=False  --eval_dir=eval/no_trans/ch1/ch2_n --data_dir=../data/chinese2 --pretrained_model_checkpoint_path=model/no_trans/ch1/ovnet.ckpt --batch_size=8 --max_images=8 --num_epoch=1 --transform=False --threshold=0.5
# python ovnet_eval.py --train_on=chinese --chinese1=False  --eval_dir=eval/no_trans/ch1/ch2   --data_dir=../data/chinese2 --pretrained_model_checkpoint_path=model/no_trans/ch1/ovnet.ckpt --batch_size=8 --max_images=8 --num_epoch=3 --transform=True  --threshold=0.5
# python ovnet_eval.py --train_on=line --eval_dir=eval/no_trans/ch1/line --pretrained_model_checkpoint_path=model/no_trans/ch1/ovnet.ckpt --batch_size=8 --max_images=8 --num_epoch=1 --threshold=0.5
# python ovnet_eval.py --train_on=sketch --eval_dir=eval/no_trans/ch1/sketch_n --data_dir=../data/sketch --pretrained_model_checkpoint_path=model/no_trans/ch1/ovnet.ckpt --batch_size=8 --max_images=8 --num_epoch=1 --transform=False --threshold=0.5
# python ovnet_eval.py --train_on=sketch --eval_dir=eval/no_trans/ch1/sketch   --data_dir=../data/sketch --pretrained_model_checkpoint_path=model/no_trans/ch1/ovnet.ckpt --batch_size=8 --max_images=8 --num_epoch=3 --transform=True  --threshold=0.5

#  ch1: trained on trans
# python ovnet_eval.py --train_on=chinese --chinese1=True   --eval_dir=eval/trans/ch1/ch1_n --data_dir=../data/chinese1 --pretrained_model_checkpoint_path=model/trans/ch1/ovnet.ckpt --batch_size=8 --max_images=8 --num_epoch=1 --transform=False --threshold=0.5
# python ovnet_eval.py --train_on=chinese --chinese1=True   --eval_dir=eval/trans/ch1/ch1   --data_dir=../data/chinese1 --pretrained_model_checkpoint_path=model/trans/ch1/ovnet.ckpt --batch_size=8 --max_images=8 --num_epoch=3 --transform=True  --threshold=0.5
# python ovnet_eval.py --train_on=chinese --chinese1=False  --eval_dir=eval/trans/ch1/ch2_n --data_dir=../data/chinese2 --pretrained_model_checkpoint_path=model/trans/ch1/ovnet.ckpt --batch_size=8 --max_images=8 --num_epoch=1 --transform=False --threshold=0.5
# python ovnet_eval.py --train_on=chinese --chinese1=False  --eval_dir=eval/trans/ch1/ch2   --data_dir=../data/chinese2 --pretrained_model_checkpoint_path=model/trans/ch1/ovnet.ckpt --batch_size=8 --max_images=8 --num_epoch=3 --transform=True  --threshold=0.5
# python ovnet_eval.py --train_on=line --eval_dir=eval/trans/ch1/line --pretrained_model_checkpoint_path=model/trans/ch1/ovnet.ckpt --batch_size=8 --max_images=8 --num_epoch=1 --threshold=0.5
# python ovnet_eval.py --train_on=sketch --eval_dir=eval/trans/ch1/sketch_n --data_dir=../data/sketch --pretrained_model_checkpoint_path=model/trans/ch1/ovnet.ckpt --batch_size=8 --max_images=8 --num_epoch=1 --transform=False --threshold=0.5
# python ovnet_eval.py --train_on=sketch --eval_dir=eval/trans/ch1/sketch   --data_dir=../data/sketch --pretrained_model_checkpoint_path=model/trans/ch1/ovnet.ckpt --batch_size=8 --max_images=8 --num_epoch=3 --transform=True  --threshold=0.5

#  ch2: trained on no trans
# python ovnet_eval.py --train_on=chinese --chinese1=True   --eval_dir=eval/no_trans/ch2/ch1_n --data_dir=../data/chinese1 --pretrained_model_checkpoint_path=model/no_trans/ch2/ovnet.ckpt --batch_size=8 --max_images=8 --num_epoch=1 --transform=False --threshold=0.5
# python ovnet_eval.py --train_on=chinese --chinese1=True   --eval_dir=eval/no_trans/ch2/ch1   --data_dir=../data/chinese1 --pretrained_model_checkpoint_path=model/no_trans/ch2/ovnet.ckpt --batch_size=8 --max_images=8 --num_epoch=3 --transform=True  --threshold=0.5
# python ovnet_eval.py --train_on=chinese --chinese1=False  --eval_dir=eval/no_trans/ch2/ch2_n --data_dir=../data/chinese2 --pretrained_model_checkpoint_path=model/no_trans/ch2/ovnet.ckpt --batch_size=8 --max_images=8 --num_epoch=1 --transform=False --threshold=0.5
# python ovnet_eval.py --train_on=chinese --chinese1=False  --eval_dir=eval/no_trans/ch2/ch2   --data_dir=../data/chinese2 --pretrained_model_checkpoint_path=model/no_trans/ch2/ovnet.ckpt --batch_size=8 --max_images=8 --num_epoch=3 --transform=True  --threshold=0.5
# python ovnet_eval.py --train_on=line --eval_dir=eval/no_trans/ch2/line --pretrained_model_checkpoint_path=model/no_trans/ch2/ovnet.ckpt --batch_size=8 --max_images=8 --num_epoch=1 --threshold=0.5
# python ovnet_eval.py --train_on=sketch --eval_dir=eval/no_trans/ch2/sketch_n --data_dir=../data/sketch --pretrained_model_checkpoint_path=model/no_trans/ch2/ovnet.ckpt --batch_size=8 --max_images=8 --num_epoch=1 --transform=False --threshold=0.5
# python ovnet_eval.py --train_on=sketch --eval_dir=eval/no_trans/ch2/sketch   --data_dir=../data/sketch --pretrained_model_checkpoint_path=model/no_trans/ch2/ovnet.ckpt --batch_size=8 --max_images=8 --num_epoch=3 --transform=True  --threshold=0.5

#  ch2: trained on trans
# python ovnet_eval.py --train_on=chinese --chinese1=True   --eval_dir=eval/trans/ch2/ch1_n --data_dir=../data/chinese1 --pretrained_model_checkpoint_path=model/trans/ch2/ovnet.ckpt --batch_size=8 --max_images=8 --num_epoch=1 --transform=False --threshold=0.5
# python ovnet_eval.py --train_on=chinese --chinese1=True   --eval_dir=eval/trans/ch2/ch1   --data_dir=../data/chinese1 --pretrained_model_checkpoint_path=model/trans/ch2/ovnet.ckpt --batch_size=8 --max_images=8 --num_epoch=3 --transform=True  --threshold=0.5
# python ovnet_eval.py --train_on=chinese --chinese1=False  --eval_dir=eval/trans/ch2/ch2_n --data_dir=../data/chinese2 --pretrained_model_checkpoint_path=model/trans/ch2/ovnet.ckpt --batch_size=8 --max_images=8 --num_epoch=1 --transform=False --threshold=0.5
# python ovnet_eval.py --train_on=chinese --chinese1=False  --eval_dir=eval/trans/ch2/ch2   --data_dir=../data/chinese2 --pretrained_model_checkpoint_path=model/trans/ch2/ovnet.ckpt --batch_size=8 --max_images=8 --num_epoch=3 --transform=True  --threshold=0.5
# python ovnet_eval.py --train_on=line --eval_dir=eval/trans/ch2/line --pretrained_model_checkpoint_path=model/trans/ch2/ovnet.ckpt --batch_size=8 --max_images=8 --num_epoch=1 --threshold=0.5
# python ovnet_eval.py --train_on=sketch --eval_dir=eval/trans/ch2/sketch_n --data_dir=../data/sketch --pretrained_model_checkpoint_path=model/trans/ch2/ovnet.ckpt --batch_size=8 --max_images=8 --num_epoch=1 --transform=False --threshold=0.5
# python ovnet_eval.py --train_on=sketch --eval_dir=eval/trans/ch2/sketch   --data_dir=../data/sketch --pretrained_model_checkpoint_path=model/trans/ch2/ovnet.ckpt --batch_size=8 --max_images=8 --num_epoch=3 --transform=True  --threshold=0.5

#  line: trained on no trans
# python ovnet_eval.py --train_on=chinese --chinese1=True   --eval_dir=eval/no_trans/line/ch1_n --data_dir=../data/chinese1 --pretrained_model_checkpoint_path=model/no_trans/line/ovnet.ckpt --batch_size=8 --max_images=8 --num_epoch=1 --transform=False --threshold=0.5
# python ovnet_eval.py --train_on=chinese --chinese1=True   --eval_dir=eval/no_trans/line/ch1   --data_dir=../data/chinese1 --pretrained_model_checkpoint_path=model/no_trans/line/ovnet.ckpt --batch_size=8 --max_images=8 --num_epoch=3 --transform=True  --threshold=0.5
# python ovnet_eval.py --train_on=chinese --chinese1=False  --eval_dir=eval/no_trans/line/ch2_n --data_dir=../data/chinese2 --pretrained_model_checkpoint_path=model/no_trans/line/ovnet.ckpt --batch_size=8 --max_images=8 --num_epoch=1 --transform=False --threshold=0.5
# python ovnet_eval.py --train_on=chinese --chinese1=False  --eval_dir=eval/no_trans/line/ch2   --data_dir=../data/chinese2 --pretrained_model_checkpoint_path=model/no_trans/line/ovnet.ckpt --batch_size=8 --max_images=8 --num_epoch=3 --transform=True  --threshold=0.5
# python ovnet_eval.py --train_on=line --eval_dir=eval/no_trans/line/line --pretrained_model_checkpoint_path=model/no_trans/line/ovnet.ckpt --batch_size=8 --max_images=8 --num_epoch=1 --threshold=0.5
# python ovnet_eval.py --train_on=sketch --eval_dir=eval/no_trans/line/sketch_n --data_dir=../data/sketch --pretrained_model_checkpoint_path=model/no_trans/line/ovnet.ckpt --batch_size=8 --max_images=8 --num_epoch=1 --transform=False --threshold=0.5
# python ovnet_eval.py --train_on=sketch --eval_dir=eval/no_trans/line/sketch   --data_dir=../data/sketch --pretrained_model_checkpoint_path=model/no_trans/line/ovnet.ckpt --batch_size=8 --max_images=8 --num_epoch=3 --transform=True  --threshold=0.5

#  sketch: trained on no trans
# python ovnet_eval.py --train_on=chinese --chinese1=True   --eval_dir=eval/no_trans/sketch/ch1_n --data_dir=../data/chinese1 --pretrained_model_checkpoint_path=model/no_trans/sketch/ovnet.ckpt --batch_size=8 --max_images=8 --num_epoch=1 --transform=False --threshold=0.5
# python ovnet_eval.py --train_on=chinese --chinese1=True   --eval_dir=eval/no_trans/sketch/ch1   --data_dir=../data/chinese1 --pretrained_model_checkpoint_path=model/no_trans/sketch/ovnet.ckpt --batch_size=8 --max_images=8 --num_epoch=3 --transform=True  --threshold=0.5
# python ovnet_eval.py --train_on=chinese --chinese1=False  --eval_dir=eval/no_trans/sketch/ch2_n --data_dir=../data/chinese2 --pretrained_model_checkpoint_path=model/no_trans/sketch/ovnet.ckpt --batch_size=8 --max_images=8 --num_epoch=1 --transform=False --threshold=0.5
# python ovnet_eval.py --train_on=chinese --chinese1=False  --eval_dir=eval/no_trans/sketch/ch2   --data_dir=../data/chinese2 --pretrained_model_checkpoint_path=model/no_trans/sketch/ovnet.ckpt --batch_size=8 --max_images=8 --num_epoch=3 --transform=True  --threshold=0.5
# python ovnet_eval.py --train_on=line --eval_dir=eval/no_trans/sketch/line --pretrained_model_checkpoint_path=model/no_trans/sketch/ovnet.ckpt --batch_size=8 --max_images=8 --num_epoch=1 --threshold=0.5
# python ovnet_eval.py --train_on=sketch --eval_dir=eval/no_trans/sketch/sketch_n --data_dir=../data/sketch --pretrained_model_checkpoint_path=model/no_trans/sketch/ovnet.ckpt --batch_size=8 --max_images=8 --num_epoch=1 --transform=False --threshold=0.5
# python ovnet_eval.py --train_on=sketch --eval_dir=eval/no_trans/sketch/sketch   --data_dir=../data/sketch --pretrained_model_checkpoint_path=model/no_trans/sketch/ovnet.ckpt --batch_size=8 --max_images=8 --num_epoch=3 --transform=True  --threshold=0.5

# #  sketch: trained on trans
# python ovnet_eval.py --train_on=chinese --chinese1=True   --eval_dir=eval/trans/sketch/ch1_n --data_dir=../data/chinese1 --pretrained_model_checkpoint_path=model/trans/sketch/ovnet.ckpt --batch_size=8 --max_images=8 --num_epoch=1 --transform=False --threshold=0.5
# python ovnet_eval.py --train_on=chinese --chinese1=True   --eval_dir=eval/trans/sketch/ch1   --data_dir=../data/chinese1 --pretrained_model_checkpoint_path=model/trans/sketch/ovnet.ckpt --batch_size=8 --max_images=8 --num_epoch=3 --transform=True  --threshold=0.5
# python ovnet_eval.py --train_on=chinese --chinese1=False  --eval_dir=eval/trans/sketch/ch2_n --data_dir=../data/chinese2 --pretrained_model_checkpoint_path=model/trans/sketch/ovnet.ckpt --batch_size=8 --max_images=8 --num_epoch=1 --transform=False --threshold=0.5
# python ovnet_eval.py --train_on=chinese --chinese1=False  --eval_dir=eval/trans/sketch/ch2   --data_dir=../data/chinese2 --pretrained_model_checkpoint_path=model/trans/sketch/ovnet.ckpt --batch_size=8 --max_images=8 --num_epoch=3 --transform=True  --threshold=0.5
# python ovnet_eval.py --train_on=line --eval_dir=eval/trans/sketch/line --pretrained_model_checkpoint_path=model/trans/sketch/ovnet.ckpt --batch_size=8 --max_images=8 --num_epoch=1 --threshold=0.5
# python ovnet_eval.py --train_on=sketch --eval_dir=eval/trans/sketch/sketch_n --data_dir=../data/sketch --pretrained_model_checkpoint_path=model/trans/sketch/ovnet.ckpt --batch_size=8 --max_images=8 --num_epoch=1 --transform=False --threshold=0.5
# python ovnet_eval.py --train_on=sketch --eval_dir=eval/trans/sketch/sketch   --data_dir=../data/sketch --pretrained_model_checkpoint_path=model/trans/sketch/ovnet.ckpt --batch_size=8 --max_images=8 --num_epoch=3 --transform=True  --threshold=0.5



#python ovnet_eval.py --train_on=chinese --chinese1=True  --eval_dir=eval/ch1_wo --data_dir=../data/chinese1 --pretrained_model_checkpoint_path=model/no_trans/ch1/ovnet.ckpt-50000 --batch_size=8 --max_images=8 --num_epoch=1 --transform=False --threshold=0.5
#python ovnet_eval.py --train_on=chinese --chinese1=True  --eval_dir=eval/ch1_w --data_dir=../data/chinese1 --pretrained_model_checkpoint_path=model/no_trans/ch1/ovnet.ckpt-50000 --batch_size=8 --max_images=8 --num_epoch=3 --transform=True --threshold=0.5
#python ovnet_eval.py --train_on=chinese --chinese1=False  --eval_dir=eval/ch2_wo --data_dir=../data/chinese2 --pretrained_model_checkpoint_path=model/r1.0_win_no_trans/ch1/ovnet.ckpt-50000 --batch_size=8 --max_images=8 --num_epoch=1 --transform=False --threshold=0.5
#python ovnet_eval.py --train_on=chinese --chinese1=False  --eval_dir=eval/ch2_w --data_dir=../data/chinese2 --pretrained_model_checkpoint_path=model/r1.0_win_no_trans/ch1/ovnet.ckpt-50000 --batch_size=8 --max_images=8 --num_epoch=3 --transform=True --threshold=0.5
#python ovnet_eval.py --train_on=line --eval_dir=eval/line --data_dir=../data/line --pretrained_model_checkpoint_path=model/r1.0_win_no_trans/ch1/ovnet.ckpt --batch_size=8 --max_images=8 --num_epoch=1 --transform=False --threshold=0.5

# python ovnet_eval.py --train_on=chinese --chinese1=False --eval_dir=eval/ch2 --data_dir=../data/chinese2 --pretrained_model_checkpoint_path=log/ch1/ovnet.ckpt --batch_size=8 --max_images=8 --num_epoch=1 --transform=False --threshold=0.5
# python ovnet_eval.py --train_on=line --eval_dir=eval/line --data_dir=../data/line --pretrained_model_checkpoint_path=log/ch1/ovnet.ckpt --batch_size=8 --max_images=8 --num_epoch=1 --transform=False --threshold=0.5

# # 18-02-17 Sat., IoU cross eval
# python ovnet_eval.py --train_on=sketch --eval_dir=eval/sketch --data_dir=../data/sketch --pretrained_model_checkpoint_path=log/sketch/ovnet.ckpt --batch_size=8 --max_images=8 --num_epoch=1 --transform=False --threshold=0.5
# python ovnet_eval.py --train_on=hand --eval_dir=eval/hand --data_dir=../data/hand --pretrained_model_checkpoint_path=log/hand/ovnet.ckpt --batch_size=8 --max_images=8 --num_epoch=1 --transform=False --threshold=0.5


# # 12-02-17 Sun., IoU cross eval
# python ovnet_eval.py --train_on=chinese --chinese1=False --eval_dir=eval/ch2 --data_dir=../data/chinese2 --pretrained_model_checkpoint_path=log/ch1/ovnet.ckpt  --batch_size=8 --max_images=8 --num_epoch=1 --transform=False --threshold=0.5
# python ovnet_eval.py --train_on=line --eval_dir=eval/line --data_dir=../data/line --pretrained_model_checkpoint_path=log/ch1/ovnet.ckpt --batch_size=8 --max_images=8 --num_epoch=1 --transform=False --threshold=0.5

# python ovnet_eval.py --train_on=chinese --chinese1=True --eval_dir=eval/ch1 --data_dir=../data/chinese1 --pretrained_model_checkpoint_path=log/ch2/ovnet.ckpt --batch_size=8 --max_images=8 --num_epoch=1 --transform=False --threshold=0.5
# python ovnet_eval.py --train_on=line --eval_dir=eval/line --data_dir=../data/line --pretrained_model_checkpoint_path=log/ch2/ovnet.ckpt --batch_size=8 --max_images=8 --num_epoch=1 --transform=False --threshold=0.5

# python ovnet_eval.py --train_on=chinese --chinese1=True --eval_dir=eval/ch1 --data_dir=../data/chinese1 --pretrained_model_checkpoint_path=log/line/ovnet.ckpt --batch_size=8 --max_images=8 --num_epoch=1 --transform=False --threshold=0.5
# python ovnet_eval.py --train_on=chinese --chinese1=False --eval_dir=eval/ch2 --data_dir=../data/chinese2 --pretrained_model_checkpoint_path=log/line/ovnet.ckpt  --batch_size=8 --max_images=8 --num_epoch=1 --transform=False --threshold=0.5

# # 12-02-17 Sun., IoU eval with threshold 0.5-0.65: 0.97, 0.45: 0.968, 0.7: 0.969
# python ovnet_eval.py --train_on=chinese --chinese1=True --eval_dir=eval/ch1 --data_dir=../data/chinese1 --pretrained_model_checkpoint_path=log/ch1/ovnet.ckpt --batch_size=8 --max_images=8 --num_epoch=3 --transform=True --threshold=0.5
# python ovnet_eval.py --train_on=chinese --chinese1=False --eval_dir=eval/ch2 --data_dir=../data/chinese2 --pretrained_model_checkpoint_path=log/ch2/ovnet.ckpt  --batch_size=8 --max_images=8 --num_epoch=3 --transform=True --threshold=0.5
# python ovnet_eval.py --train_on=line --eval_dir=eval/line --data_dir=../data/line --pretrained_model_checkpoint_path=log/line/ovnet.ckpt --batch_size=8 --max_images=8 --num_epoch=3 --transform=True --threshold=0.5
# no trans
# python ovnet_eval.py --train_on=chinese --chinese1=True --eval_dir=eval/ch1 --data_dir=../data/chinese1 --pretrained_model_checkpoint_path=log/ch1/ovnet.ckpt --batch_size=8 --max_images=8 --num_epoch=1 --transform=False --threshold=0.5
# python ovnet_eval.py --train_on=chinese --chinese1=False --eval_dir=eval/ch2 --data_dir=../data/chinese2 --pretrained_model_checkpoint_path=log/ch2/ovnet.ckpt  --batch_size=8 --max_images=8 --num_epoch=1 --transform=False --threshold=0.5
# python ovnet_eval.py --train_on=line --eval_dir=eval/line --data_dir=../data/line --pretrained_model_checkpoint_path=log/line/ovnet.ckpt --batch_size=8 --max_images=8 --num_epoch=1 --transform=False --threshold=0.5

# 11-02-17 Sat., IoU eval
# python ovnet_eval.py --train_on=chinese --chinese1=True --eval_dir=eval/ch1 --data_dir=../data/chinese1 --pretrained_model_checkpoint_path=log/ch1/ovnet.ckpt --batch_size=8 --max_images=8 --num_epoch=1 --transform=False --threshold=0.95 # -> 0.937
# python ovnet_eval.py --train_on=chinese --chinese1=False --eval_dir=eval/ch2 --data_dir=../data/chinese2 --pretrained_model_checkpoint_path=log/ch2/ovnet.ckpt  --batch_size=8 --max_images=8 --num_epoch=1 --transform=False
# python ovnet_eval.py --train_on=line --eval_dir=eval/line --data_dir=../data/line --pretrained_model_checkpoint_path=log/line/ovnet.ckpt --batch_size=8 --max_images=8 --num_epoch=1 --transform=False

# # 11-02-17 Sat., second eval (failed ones)
# python ovnet_eval.py --train_on=line --eval_dir=eval/line --data_dir=../data/line --pretrained_model_checkpoint_path=log/line/ovnet.ckpt --batch_size=8 --max_images=8 --num_epoch=10 --transform=False
# python ovnet_eval.py --train_on=sketch --eval_dir=eval/sketch --data_dir=../data/sketch --pretrained_model_checkpoint_path=log/line/ovnet.ckpt --batch_size=8 --max_images=8 --num_epoch=10 --transform=False
# python ovnet_eval.py --train_on=hand --eval_dir=eval/hand --data_dir=../data/hand --pretrained_model_checkpoint_path=log/line/ovnet.ckpt --batch_size=8 --max_images=8 --num_epoch=10 --transform=False

# # 03-02-17 Fri., first eval
# python ovnet_eval.py --train_on=chinese --chinese1=True --eval_dir=eval/ch1 --data_dir=../data/chinese1 --pretrained_model_checkpoint_path=log/ch1/ovnet.ckpt --batch_size=8 --max_images=8 --num_epoch=10 --transform=False
# python ovnet_eval.py --train_on=chinese --chinese1=False --eval_dir=eval/ch2 --data_dir=../data/chinese2 --pretrained_model_checkpoint_path=log/ch2/ovnet.ckpt  --batch_size=8 --max_images=8 --num_epoch=10 --transform=False