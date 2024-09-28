# train Teacher
python main.py --gpus 2 --path base_vit --store_path aug_patchmask --augment --aug_type patchmask --batch_size 64 --hash_bit 256 --dataset UCMD --epochs=300
# train Student
#python main.py --gpus 2,3 --path base_distill --store_path aug_patchmask --augment --aug_type patchmask --batch_size 128 --hash_bit  256 --dataset UCMD --epochs=300 --teacher "logs/base_vit/***/**.pt"

