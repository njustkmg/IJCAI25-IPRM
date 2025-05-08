# Towards Equilibrium: An Instantaneous Probe-and-Rebalance Multimodal Learning Approach
 This is the official implementation of the IPRM approach proposed by our paper titled ''Towards Equilibrium: An Instantaneous Probe-and-Rebalance Multimodal Learning Approach''.

 **Paper Title: Towards Equilibrium: An Instantaneous Probe-and-Rebalance Multimodal Learning Approach**
 ## Code instruction
 ### For training a bimodal dataset, taking CREMA-D as an example.
```python
python main_av.py --dataset CREMAD --gpu_ids 0 --batch_size 64 --epoch 100 --train --alpha 0.8 --ckpt_path log_cd --saved_model_name CREMAD
```
### For training a trimodal dataset with Tri-CLS style, taking NVGesture as an example.
```python
python main_trimodal.py--dataset NVGesture --gpu_ids 0 --batch_size 2 --epoch 100 --train --alpha 0.8 --mixup_method tri_cls --ckpt_path log_cd --saved_model_name NVGesture 
```
### For training a trimodal dataset with Single-CLS style, taking NVGesture as an example.
```python
python main_trimodal.py--dataset NVGesture --gpu_ids 0 --batch_size 2 --epoch 100 --train --alpha 0.8 --mixup_method single_cls --ckpt_path log_cd --saved_model_name NVGesture 
```
