#!/bin/zsh

#SBATCH --job-name=h5test
#SBATCH --output=h5test.out
#SBATCH --partition=medusa
#SBATCH --nodes=2
#SBATCH --time=00:30:00

# set xe

# load modules
# module load gcc openmpi hdf5 python


# check if build dir exists, if not then create it
#DIR="/lustre/home/mjcole/vascpp/h5test/build"
DIR="/work/maxwell/h5test/build"
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

#mpirun /usr/bin/time -v  ./test
srun ./test
