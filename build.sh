#!/bin/zsh

#SBATCH --job-name=h5test
#SBATCH --output=itest3200.out
#SBATCH --ntasks-per-node=1
#SBATCH -N 128
#SBATCH --time=00:30:00
#SBATCH -p all-nodes

# set xe

# load modules
# module load gcc openmpi hdf5 python


# check if build dir exists, if not then create it
DIR="/lustre/home/mjcole/vascpp/h5test/build"
if [ -d "$DIR" ]; then 
    cd build
else
    echo "Directory 'build' not found"
    echo "Creating directory 'build' "
    mkdir build && cd build
fi

# build with cmake
cmake ..
echo "Building . . ."
cmake --build .

mpirun /usr/bin/time -v  ./test
