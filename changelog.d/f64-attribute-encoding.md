- Fixed bytecode encoding of negative float64 attributes, including the `-inf`
identity used by float64 `max` and `argmax` reductions. These attributes could
previously cause compilation failures or be decoded with an incorrect sign.
