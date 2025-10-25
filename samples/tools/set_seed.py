import random

import numpy as np
import torch



def set_seed(seed: int) -> None:
    """
    Pytorch, NumPyのシード値を固定します．これによりモデル学習の再現性を担保できます．

    Parameters
    ----------
    seed : int
        シード値．
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False