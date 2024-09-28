# train Teacher
python main.py --gpus 2 --path base_vit --store_path aug_patchmask --augment --aug_type patchmask --batch_size 64 --hash_bit 256 --dataset UCMD --epochs=300