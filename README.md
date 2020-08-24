# For test on Nuscenes
```
$ cd ~/workspace/bts
$ python Exp_trainNuscenes/bts_nuscenes_test.py --model_name bts_nuscenes --encoder densenet161_bts --dataset kitti --data_path pathToNuscenes --filenames_file ../train_test_inputs/nuscenes_test_files_with_gt.txt --input_height 352 --input_width 704 --max_depth 80 --min_depth_eval 1e-3 --max_depth_eval 80 --checkpoint_path pathToCheckPoint
```
Then model can be downloaded from [Here](https://drive.google.com/file/d/1rY4aZOKUtVHcWUjN1W_k5FZLr6UPdnIZ/view?usp=sharing)

You should see outputs like this:
```
Raw png files reading done
Evaluating 654 files
GT files reading done
0 GT files missing
Computing errors
     d1,      d2,      d3,   AbsRel,   SqRel,    RMSE,    RMSElog
  0.922,   0.974,   0.988,   0.080,    0.651,    4.313,   0.146
```