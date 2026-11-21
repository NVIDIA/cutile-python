- Fix compilation cache eviction leaving the cache above `CUDA_TILE_CACHE_SIZE`
  when entry sizes differ: the oldest entries are now evicted until the
  remaining size is within the limit.
