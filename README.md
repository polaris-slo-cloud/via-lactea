

Training 

```
 conda activate snnet
 
./SN-Net/stitching_resnet_swin/distributed_train.sh 1 ./dataset/wildfire_dataset -b 128 --stitch_config configs/resnet18_swin_ti.json --sched cosine --epochs 30 --lr 0.05   --amp --remode pixel --reprob 0.6 --aa rand-m9-mstd0.5-inc1 --resplit --split-bn -j 10 --dist-bn reduce
```

Results

```
./SN-Net/stitching_resnet_swin/output
```
Training model head
```
python3 train_head_and_spread.py --ckpt output/train/20251127-105806-resnet50-224/model_best.pth.tar --stitch_id 1  --train-root ../../dataset/wildfire_canada/train/ --val-root  ../../dataset/wildfire_canada/test/ --num-classes 2  --epochs 5 --batch-size 256 --workers 8 --amp
```

Evaluate stitching id
```
python3 eval_stitch_ids.py  --ckpt stitched_head_ft_wildfire.pth.tar  --val-root ../../dataset/cifar-10-imagenet/val/ --img-size 224  --batch-size 256  --workers 8 --no-center-crop --amp 
```
Adding noise to val dataset
```
--cifar10-transforms --no-center-crop --amp --corrupt gauss_noise --severity 1 --stitch-noise-std 1
```