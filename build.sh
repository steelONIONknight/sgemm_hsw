if [ "$1" == "arm" ];then
  gcc -O3 -c main.c
  gcc -O3 -o sgemm_l1d main.o
else
  as -o sgemm_kernel_x64_fma.o sgemm_kernel_x64_fma.S
  gcc -O3 -c main.c
  gcc -O3 -pthread -o sgemm_l1d sgemm_kernel_x64_fma.o main.o
fi

