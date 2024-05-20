import os
#search the generator
os.system('''CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 python ./EGANS/clswgan_G_search.py \
--dataset CUB --dataroot ./EGANS/data/xlsa17/data/ --geo_dir ./EGANS/output/genotypes/ \
--syn_num 400 --preprocessing --batch_size 64 --resSize 2048 \
--attSize 312 --nz 312 --warmup_nepoch 2 --critic_iter 5 --lambda1 10  --cls_weight 0.2 \
--lr 0.0001 --classifier_lr 0.001 --cuda --image_embedding res101 \
--nclass_all 200 --class_embedding att --nepoch 80 --epochs 100 \
--num_individual 50 --num_train 2 --regular_weight 1.0 ''')

#search the discriminator
os.system('''CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 python ./EGANS/clswgan_D_search.py \
--dataset CUB --dataroot ./EGANS/data/xlsa17/data/ --geo_dir ./EGANS/output/genotypes/ \
--syn_num 400 --gzsl --preprocessing --batch_size 64 --resSize 2048 \
--attSize 312 --nz 312 --warmup_nepoch 2 --critic_iter 5 --lambda1 10  --cls_weight 0.2 \
--lr 0.0001 --classifier_lr 0.001 --cuda --image_embedding res101 \
--nclass_all 200 --class_embedding att --nepoch 100 --epochs 100 \
--num_individual 50 --num_train 2 --regular_weight 3.0 ''')

#retrain the model
os.system('''CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 python./EGANS/clswgan_retrain.py \
--dataset CUB --dataroot ./EGANS/data/xlsa17/data/ --geo_dir ./EGANS/output/genotypes/ \
--syn_num 400 --preprocessing --batch_size 64 --resSize 2048 \
--attSize 312 --nz 312 --critic_iter 5 --lambda1 10  --cls_weight 0.2 \
--lr 0.0001 --classifier_lr 0.001 --cuda --image_embedding res101 \
--nclass_all 200 --class_embedding att --nepoch 150 --slow 1 ''')