# python vectorize_png_big.py --test_dir=result/no_overlap/fidelity_128 --data_dir=data/fidelity/png_128 --num_test_files=15 --image_height=128 --image_width=128 --pathnet_ckpt=pathnet/model/no_trans_64/line/pathnet.ckpt-50000 --find_overlap=False --batch_size=512 --neighbor_sigma=16
# python vectorize_png_big.py --test_dir=result/no_overlap/fidelity_256  --data_dir=data/fidelity/png_256 --num_test_files=15 --image_height=256  --image_width=256  --pathnet_ckpt=pathnet/model/no_trans_64/line/pathnet.ckpt-50000 --find_overlap=False --batch_size=512 --neighbor_sigma=16
# python vectorize_png_big.py --test_dir=result/no_overlap/fidelity_512  --data_dir=data/fidelity/png_512 --num_test_files=15 --image_height=512  --image_width=512  --pathnet_ckpt=pathnet/model/no_trans_64/line/pathnet.ckpt-50000 --find_overlap=False --batch_size=128 --neighbor_sigma=32

# python vectorize_png_big.py --test_dir=result/overlap/fidelity_128  --data_dir=data/fidelity/png_128 --num_test_files=15 --image_height=128  --image_width=128  --pathnet_ckpt=pathnet/model/no_trans_64/line/pathnet.ckpt-50000 --find_overlap=False --batch_size=512 --neighbor_sigma=16 --ovnet_ckpt=ovnet/model/no_trans_64/ch1/ovnet.ckpt-50000 --find_overlap=True 
python vectorize_png_big.py --test_dir=result/overlap/fidelity_256_line_ch1  --data_dir=data/fidelity/png_256 --num_test_files=15 --image_height=256  --image_width=256  --pathnet_ckpt=pathnet/model/no_trans_64/line/pathnet.ckpt-50000 --find_overlap=False --batch_size=512 --neighbor_sigma=16 --ovnet_ckpt=ovnet/model/no_trans_64/ch1/ovnet.ckpt-50000 --find_overlap=True --threshold=0.5
# python vectorize_png_big.py --test_dir=result/overlap/fidelity_256_line_line2  --data_dir=data/fidelity/png_256 --num_test_files=15 --image_height=256  --image_width=256  --pathnet_ckpt=pathnet/model/no_trans_64/line/pathnet.ckpt-50000 --find_overlap=False --batch_size=512 --neighbor_sigma=16 --ovnet_ckpt=ovnet/model/no_trans_64/line/ovnet.ckpt-50000 --find_overlap=True

# python vectorize_png.py --test_dir=result/no_overlap/fidelity_256 --data_dir=data/fidelity/png_256 --num_test_files=15 --image_height=256 --image_width=256 --pathnet_ckpt=pathnet/model/no_trans_64/line/pathnet.ckpt-50000 --ovnet_ckpt=ovnet/model/no_trans_64/line/ovnet.ckpt-50000 --find_overlap=False --batch_size=32
# python vectorize_png.py --test_dir=result/no_overlap/fidelity_128 --data_dir=data/fidelity/png_128 --num_test_files=15 --image_height=128 --image_width=128 --pathnet_ckpt=pathnet/model/no_trans_64/line/pathnet.ckpt-50000 --ovnet_ckpt=ovnet/model/no_trans_64/line/ovnet.ckpt-50000 --find_overlap=False --batch_size=128
# python vectorize_png.py --test_dir=result/no_overlap/fidelity_64 --data_dir=data/fidelity/png_64 --num_test_files=15 --image_height=64 --image_width=64 --pathnet_ckpt=pathnet/model/no_trans_64/line/pathnet.ckpt-50000 --ovnet_ckpt=ovnet/model/no_trans_64/line/ovnet.ckpt-50000 --find_overlap=False --batch_size=512