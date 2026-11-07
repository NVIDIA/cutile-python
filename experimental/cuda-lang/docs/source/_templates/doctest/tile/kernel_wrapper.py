import cuda.tile as ct
import torch

@ct.kernel
def kernel():
    # begin-snippet
    {{body}}
    # end-snippet


torch.cuda.init()
ct.launch(torch.cuda.current_stream(), (1,), kernel, ())
torch.cuda.synchronize()
