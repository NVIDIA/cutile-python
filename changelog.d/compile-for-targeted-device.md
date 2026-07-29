- Fixed ``ct.launch()`` compiling kernels for CUDA device 0 rather than for the device being
  launched on. Kernels are now compiled for the launch device's architecture. Devices with the
  same architecture can still share a compiled kernel.
