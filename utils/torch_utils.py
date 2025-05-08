import torch
import numpy as np
import random
from typing import Tuple

def get_device_and_num_gpus() -> Tuple[torch.device, int]:
    """
    PyTorchの計算に使用するデバイスを決定し、それと利用可能なGPUの数を返します。

    関数はCUDAが利用可能かどうかを確認し、利用可能であればGPUをデバイスとして使用し、利用可能なGPUの数を返します。
    CUDAが利用できない場合は、CPUをデバイスとして使用し、GPUの数として0を返します。

    戻り値:
        使用するデバイスと利用可能なGPUの数を含むタプル。
    """
    if torch.cuda.is_available():
        # CUDAが利用可能であれば、GPUをデバイスとして使用
        device = torch.device("cuda")
        # 利用可能なGPUの数を取得
        num_gpus = torch.cuda.device_count()
    else:
        # CUDAが利用できない場合は、CPUをデバイスとして使用
        device = torch.device("cpu")
        # 利用可能なGPUはない
        num_gpus = 0
    
    # デバイスと利用可能なGPUの数を返す
    return device, num_gpus


def set_seed(seed: int) -> None:
    """
    ランダム数生成器のシードを設定します。

    この関数は以下のランダム数生成器のシードを設定します:
        - Pythonの組み込み`random`モジュール
        - NumPyの`numpy.random`モジュール
        - PyTorchの`torch.manual_seed`および`torch.cuda.manual_seed_all`
        - CUDNNの`torch.backends.cudnn.deterministic`フラグ

    シードを設定することで、実験の結果が再現可能になります。
    """
    # Pythonの組み込みrandomモジュールのシードを設定
    random.seed(seed)
    # NumPyのnumpy.randomモジュールのシードを設定
    np.random.seed(seed)
    # PyTorchのtorch.manual_seedおよびtorch.cuda.manual_seed_allのシードを設定
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # CUDNNのtorch.backends.cudnn.deterministicフラグを設定
    torch.backends.cudnn.deterministic = True
