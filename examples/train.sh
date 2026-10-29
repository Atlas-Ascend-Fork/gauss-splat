export ASCEND_RT_VISIBLE_DEVICES=0
export TASK_QUEUE_ENABLE=2

NPU_VISIBLE_DEVICES=0 python train.py --data_dir /home/360_v2/garden --result_dir results/benchmark/garden --data_factor 4 --depth_loss --tile_size 32