

#include <random>
#include <iostream>
#include <mpi.h>
#include <hdf5.h>

using namespace std;

void collective(long long rows, long long cols, long long localRows, double* arr);
void independent(long long rows, long long cols, long long localRows, double* arr);

int main (){
	MPI_Init(NULL, NULL);
	int mpiSize, mpiRank;
	MPI_Comm_size(MPI_COMM_WORLD, &mpiSize);
	MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);
	double TotalTime = MPI_Wtime();

	// create 2d array of random numbers
	double time = MPI_Wtime();
	// long long rows = mpiSize*125000;
	long long rows = mpiSize*250000;
	long long cols = 12500;
	long long localRows = rows/mpiSize;
	std::uniform_real_distribution<double> unif(0.00001, 10);
	std::default_random_engine re;
	double* arr = new double[localRows * cols];
	//int num = localRows*cols*mpiRank;
	for(int i = 0; i < localRows; i++) {
		for(int j = 0; j < cols; j++) {
			*(arr + i * cols + j) = unif(re);
			// *(arr + i * cols + j) = num;
			// num++;
		}
	}
	if (!mpiRank) { cout << "Create matrix time: " << (MPI_Wtime() - time) << " seconds" << endl; }	

	// perform collective write to hdf5
	//collective(rows, cols, localRows, arr);

	// perform independent write to hdf5
	independent(rows, cols, localRows, arr);

	// end app
	MPI_Barrier(MPI_COMM_WORLD);
	if (!mpiRank) { cout << "Total time: " << (MPI_Wtime() - TotalTime) << " seconds\n\n" << endl; }
	MPI_Finalize();
	return 0;
}


// function to perform collective write to one hdf5 file
void collective(long long rows, long long cols, long long localRows, double* arr) {
	int mpiSize, mpiRank;
	MPI_Comm_size(MPI_COMM_WORLD, &mpiSize);
	MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);
	double time = MPI_Wtime();

	if (!mpiRank) std::cout << "\nPerforming COLLECTIVE Write\n" << std::endl;

	// create FAPL
	time = MPI_Wtime();
	MPI_Info info;
    MPI_Info_create(&info);
	hid_t memFaplId = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_mpio(memFaplId, MPI_COMM_WORLD, info);
	if (!mpiRank) { cout << "- Create fapl time: " << (MPI_Wtime() - time) << " seconds" << endl; }

	// create file
	time = MPI_Wtime();
	std::string filename = "h5test.h5";
	hid_t fcplId = H5Pcreate(H5P_FILE_CREATE);
    H5Pset_sym_k(fcplId, 16, 8);
	hid_t file_id = H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, fcplId, memFaplId);
	H5Pclose(memFaplId);
    H5Pclose(fcplId);
	if (!mpiRank) { cout << "- Create file time: " << (MPI_Wtime() - time) << " seconds" << endl; }

	// create group
	time = MPI_Wtime();
	hid_t group_id;
	std::string group = "example";
    group_id = H5Gcreate(file_id, group.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Gclose(group_id);
	if (!mpiRank) { cout << "- Create group time: " << (MPI_Wtime() - time) << " seconds" << endl; }

	// create dataset
	time = MPI_Wtime();
	std::string path = "example/ds";
	hid_t dataspaceID, datasetID;
    herr_t status;
    hid_t dcpl;
    hsize_t dims[2];
    dims[0] = rows;
    dims[1] = cols;
    dataspaceID = H5Screate_simple(2, dims, NULL);
    dcpl = H5Pcreate(H5P_DATASET_CREATE);
	H5Pset_alloc_time(dcpl, H5D_ALLOC_TIME_EARLY);
    H5Pset_fill_time(dcpl, H5D_FILL_TIME_NEVER);
    datasetID = H5Dcreate2(file_id, path.c_str(), H5T_NATIVE_DOUBLE, dataspaceID, H5P_DEFAULT, dcpl, H5P_DEFAULT);
    status = H5Pclose(dcpl);
	if (!mpiRank) { cout << "- Create dataset time: " << (MPI_Wtime() - time) << " seconds" << endl; }


	// write to file
	long long startRowIndex = mpiRank * localRows;
	hid_t dtpl = H5Pcreate(H5P_DATASET_XFER);
    H5Pset_dxpl_mpio(dtpl, H5FD_MPIO_COLLECTIVE);
    hsize_t start[2] = {0}, count[2] = {0}, strideAndBlocks[2] = {1, 1};
	int ndims = 2;
    count[0] = localRows;
    count[1] = cols;
    start[0] = startRowIndex;
    start[1] = 0;
    hid_t memspaceId = H5Screate_simple(ndims, count, NULL);
    H5Sselect_hyperslab(dataspaceID, H5S_SELECT_SET, start, NULL, count, NULL);
    H5Dwrite(datasetID, H5T_NATIVE_DOUBLE, memspaceId, dataspaceID, dtpl, arr);


    H5Sclose(dataspaceID);
    H5Sclose(memspaceId);
    H5Pclose(dtpl);
	H5Dclose(datasetID);
	MPI_Barrier(MPI_COMM_WORLD);
	if (!mpiRank) { cout << "-- Write arrays time: " << (MPI_Wtime() - time) << " seconds" << endl; }

	status = H5Fclose(file_id);
}


// function to perform independent write to N=mpiSize hdf5 files
void independent(long long rows, long long cols, long long localRows, double* arr) {
	int mpiSize, mpiRank;
	MPI_Comm_size(MPI_COMM_WORLD, &mpiSize);
	MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);
	double time = MPI_Wtime();

	if (!mpiRank) std::cout << "\nPerforming INDEPENDENT Write\n" << std::endl;

	// split MPI Communicator: each process gets its own COMM
	std::string var = "comm_" + std::to_string(mpiRank);
	//std::cout << mpiRank << ": " << var << std::endl;
	MPI_Comm vcomm;
	MPI_Comm_split(MPI_COMM_WORLD, mpiRank, mpiRank, &vcomm);
	int rankRank, rankSize;
	MPI_Comm_size(vcomm, &rankSize);
	MPI_Comm_rank(vcomm, &rankRank);
	//printf("WORLD RANK/SIZE: %d/%d \t Rank RANK/SIZE: %d/%d\n", mpiRank, mpiSize, rankRank, rankSize);

	// create FAPL
	time = MPI_Wtime();
	MPI_Info info;
    MPI_Info_create(&info);
	hid_t memFaplId = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_mpio(memFaplId, vcomm, info);
	if (!mpiRank) { cout << "- Create fapl time: " << (MPI_Wtime() - time) << " seconds" << endl; }

	// create file per rank
	time = MPI_Wtime();
	std::string filename = "file_" + std::to_string(mpiRank) + ".h5";
	//std::cout << mpiRank << ": " << filename << std::endl;
	hid_t fcplId = H5Pcreate(H5P_FILE_CREATE);
    H5Pset_sym_k(fcplId, 16, 8);
	hid_t file_id = H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, fcplId, memFaplId);
	H5Pclose(memFaplId);
    H5Pclose(fcplId);
	if (!mpiRank) { cout << "- Create file time: " << (MPI_Wtime() - time) << " seconds" << endl; }

	// create group
	time = MPI_Wtime();
	hid_t group_id;
	std::string group = "example";
    group_id = H5Gcreate(file_id, group.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Gclose(group_id);
	if (!mpiRank) { cout << "- Create group time: " << (MPI_Wtime() - time) << " seconds" << endl; }

	// create dataset
	time = MPI_Wtime();
	std::string path = "example/ds";
	hid_t dataspaceID, datasetID;
    herr_t status;
    hid_t dcpl;
    hsize_t dims[2];
    dims[0] = localRows;
    dims[1] = cols;
    dataspaceID = H5Screate_simple(2, dims, NULL);
    dcpl = H5Pcreate(H5P_DATASET_CREATE);
	H5Pset_alloc_time(dcpl, H5D_ALLOC_TIME_EARLY);
    H5Pset_fill_time(dcpl, H5D_FILL_TIME_NEVER);
    datasetID = H5Dcreate2(file_id, path.c_str(), H5T_NATIVE_DOUBLE, dataspaceID, H5P_DEFAULT, dcpl, H5P_DEFAULT);
    status = H5Pclose(dcpl);
	if (!mpiRank) { cout << "- Create dataset time: " << (MPI_Wtime() - time) << " seconds" << endl; }


	// write to file
	hid_t dtpl = H5Pcreate(H5P_DATASET_XFER);
    H5Pset_dxpl_mpio(dtpl, H5FD_MPIO_INDEPENDENT);
    H5Dwrite(datasetID, H5T_NATIVE_DOUBLE, H5S_ALL, dataspaceID, dtpl, arr);


    H5Sclose(dataspaceID);
    H5Pclose(dtpl);
	H5Dclose(datasetID);

	MPI_Comm_free(&vcomm);
	MPI_Barrier(MPI_COMM_WORLD);
	if (!mpiRank) { cout << "-- Write arrays time: " << (MPI_Wtime() - time) << " seconds" << endl; }

	status = H5Fclose(file_id);
}
