> Command for running(original)
```
cd /data2/chengyi/myproject
source activate cuda10
python3 main.py test --net p3d199 --tag much_seg -tf -rs --modality GRAY --num_segments 16 --lr 0.0003 --gpus 3 2 1 4
```


> Ada Command:

`python3 main.py test --net Ada 
--tag much_seg -tf -rs --modality 
Ada --num_segments 32 --lr 0.0003 --gpus 
2 3 -ld 25 --epoch 100 --loss ada_loss --bs 4`

`source activate cuda10`

`
CUDA_VISIBLE_DEVICES=3 python3 main.py test --net Ada 
--tag much_seg -tf -rs --modality 
Ada --num_segments 32 --lr 0.0003 
-ld 25 --epoch 75 --loss ada_loss --bs 3 --cp --cr --lw 1.5`

cp cr --> best acc

`
CUDA_VISIBLE_DEVICES=8 python3 main.py test --net Ada 
--tag much_seg -tf -rs --modality 
Ada --num_segments 32 --lr 0.0003 
-ld 25 --epoch 75 --loss ada_loss --bs 3 --cp --cr --rw 0.5`


try cell-hidden-ratio:

source activate cuda10

`
CUDA_VISIBLE_DEVICES=8 python3 main.py test --net Ada 
--tag much_seg -tf -rs --modality 
Ada --num_segments 32 --lr 0.0003 
-ld 25 --epoch 75 --loss ada_loss --bs 3 --chr 0.5`


`
CUDA_VISIBLE_DEVICES=8 python3 main.py test --net AdaViT --tag much_seg -tf 
-rs --modality Ada --num_segments 32 
--lr 0.0003 -ld 25 --epoch 75 --loss Cross_Entropy --bs 3
`

feature+encode+lstm:

`
CUDA_VISIBLE_DEVICES=5 python3 main.py test --net Ada --tag much_seg
-tf -rs --modality Ada --num_segments 32 --lr 0.0003 -ld 25 
--epoch 75 --loss Cross_Entropy --bs 2 --adamode transformer
`

`
CUDA_VISIBLE_DEVICES=3 python3 main.py test --net Ada 
--tag much_seg -tf -rs --modality 
Ada --num_segments 64 --lr 0.0003 
-ld 25 --epoch 75 --loss ada_loss --bs 2 --cp --cr`

```
BEST combination:
CUDA_VISIBLE_DEVICES=9 python3 main.py test --net Ada --tag adatest -tf -rs --modality Ada --num_segments 32 --lr 0.0003 -ld 25 --epoch 75 --loss ada_loss --bs 3 --cp --cr 
```