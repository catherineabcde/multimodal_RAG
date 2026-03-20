import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
print("Device count:", torch.cuda.device_count())
print("Device name:", torch.cuda.get_device_name(0))
