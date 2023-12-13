export OMP_NUM_THREADS=1

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 -m torch.distributed.launch \
--master_addr="127.0.0.2" \
--master_port=29503 \
--nproc_per_node=1 \
CAM_all.py  --path='../Wengweixiang/patches/'\
    --epoch='80'\
    --sample='20229999KF_299_inceptionv3'\
    --fold='4'\
    --mag='10'\
    --model='inceptionv3'\
    --test_limit=15\
    --extd=11\
    --save_path='../../../../../gputemp/ToWZP/CAM_CC'\
    --init_method=env://
    
ps aux | grep MILEval | awk '{print "kill -9 " $2}'| sh