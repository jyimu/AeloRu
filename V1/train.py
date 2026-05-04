
import sys
sys.path.append('..')  # 添加父目录到路径
from data.gsm8k.gsm8k import Gsm8k
import torch
from .aeloru_layer import inject_hidora