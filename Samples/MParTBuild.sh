#!/bin/bash
# Example script to build MParT once Kokkos is installed at the appropriate locations.
install_base="/home/dannys4/installs/"

for backend in SERIAL THREADS OPENMP
do
    
    if [ -d "build-"$backend ]; then 
        rm -rf "build-"$backend 
    fi 

    mkdir "build-"$backend 
    cd "build-"$backend 

    cmake -GNinja -DCMAKE_BUILD_TYPE=RELEASE -DMPART_PYTHON=OFF -DMPART_MATLAB=OFF -DMPART_JULIA=OFF -DMPART_BUILD_TESTS=OFF -DKokkos_ROOT=$install_base"KOKKOS_"$backend -DCMAKE_INSTALL_PREFIX=$install_base"MPART_"$backend ../
    ninja
    cmake --install .

    cd ../

done
