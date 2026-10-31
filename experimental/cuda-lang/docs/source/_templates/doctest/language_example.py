import cuda.lang as cl
import torch

torch.cuda.init()
stream = torch.cuda.current_stream()

{{body}}

torch.cuda.synchronize()
