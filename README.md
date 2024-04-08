# CurFormer

## Structuring
![img.png](doc/sturcturing.png)

## Feature Extraction
![img.png](doc/pfe.png)

## Group Intersection
![img.png](doc/group_intersection.png)

## Pooling & UnPooling
![img.png](doc/pool.png)



## Beginning
### train
`python tools/train.py --cfg configs/voxformer/voxformer_4x2_80e_kitti_3cls.py`

### eval
`python tools/test.py --cfg configs/voxformer/voxformer_4x2_80e_kitti_3cls.py --ckpt the_ckpt_in_output_dir`
