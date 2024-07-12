python -u train_stylesdf.py --size 1024 --batch 2 --n_sample 1 --output_dir model/stylesdf_spongebob_model \
--lr 0.0002 --frozen_gen_ckpt /home/chenzhuo/workspace/cartoonGAN/model_zoo/afhq512x512.pt \
--iter 301 --source_class "human" --target_class "SpongeBob" \
--auto_layer_k 18 --auto_layer_iters 1 --auto_layer_batch 8  --output_interval 20 \
--clip_models "/home/chenzhuo/workspace/cartoonGAN/model_zoo/ViT-B-32.pt" "/home/chenzhuo/workspace/cartoonGAN/model_zoo/ViT-B-16.pt" \
--clip_model_weights 1.0 1.0  --mixing 0.0   --save_interval 150

