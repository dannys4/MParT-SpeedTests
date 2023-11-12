#!/bin/bash

install_base="/home/dannys4/installs/"

# Build the executables for each backend type
for backend in SERIAL THREADS OPENMP
do
    
    if [ -d "build-"$backend ]; then 
        rm -rf "build-"$backend 
    fi 

    mkdir "build-"$backend 
    cd "build-"$backend 

    if [ "$backend" == "CUDA" ]; then
        cmake -DKokkos_ROOT=$install_base"KOKKOS_"$backend -DMParT_ROOT=$install_base"MPART_"$backend -DCMAKE_CXX_COMPILER="$install_base"KOKKOS_CUDA/bin/nvcc_wrapper ../
        make
    else
        cmake -DKokkos_ROOT=$install_base"KOKKOS_"$backend -DMParT_ROOT=$install_base"MPART_"$backend ../
        make
    fi 

    cd ../

done


# Run the serial tests 
cd build-SERIAL 
./SampleSpeed SERIAL 
cd ../ 

export OMP_PROC_BIND=spread
export OMP_PLACES=threads

# Run the host multithreaded backends
for backend in THREADS OPENMP
do 

    cd "build-"$backend 

    for num_threads in 2 4 8 16 32
    do 
        ./SampleSpeed $backend $num_threads
    done 

    cd ../ 
done 

# Run the CUDA backend 
# cd build-CUDA 
# ./SampleSpeed CUDA 
# cd ../

# Collect all the results in the "results" folder
now=$(date +'%m%d%Y_%H%M')
mkdir results$now

for backend in SERIAL THREADS OPENMP
do
    cp "build-"$backend/ST_CPP_*.txt results$now/
done 

lscpu > results$now/cpu_spec.txt 




