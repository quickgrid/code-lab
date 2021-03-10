## Commands

- In linux use `df -h` to get information like disk size, free space, used space, mounted on etc.

- In windows `nvidia-smi` can be found under, `C:\Program Files\NVIDIA Corporation\NVSMI`. In `colab` or `aws` jupyter notebooks add `!` before the command below. 
  ```
  nvidia-smi --format=csv --query-gpu=power.draw,utilization.gpu,fan.speed,temperature.gpu,memory.used,memory.free
  ```
