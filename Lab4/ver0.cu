#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <vector>

#define NUM_THREADS_PER_DIM 32

using namespace std;

__global__ void MatrixMul(const int* a, const int* b, int* c, int n) {
	// Compute each thread's global row and column index
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	// Iterate over row, and down column
	int tmp = 0;
	for (int k = 0; k < n; k++) {
		// Accumulate results for a single element
		tmp += a[row * n + k] * b[k * n + col];
	}

	c[row * n + col] = tmp;
}

// Check result on the CPU
void Validate(vector<int>& a, vector<int>& b, vector<int>& c, int n) {
	// For every row...
	for (int i = 0; i < n; i++) {
		// For every column...
		for (int j = 0; j < n; j++) {
			// For every element in the row-column pair
			int tmp = 0;
			for (int k = 0; k < n; k++) {
				// Accumulate the partial results
				tmp += a[i * n + k] * b[k * n + j];
			}

			// Check against the CPU result
			if (tmp != c[i * n + j]) {
				cout << "Mismatch at " << tmp << " " << c[i * n + j] << " " << i << " " << j << endl;
				return;
			}
		}
	}

	cout << "All Good!" << endl;
}

// Function for initializing matrix on CPU
void Initialize(vector<int>& a) {

	srand((unsigned int)time(NULL));

	int max_val = 1 << 7;

	for (auto i = a.begin(); i < a.end(); i++)
	{
		*i = (int)rand() % max_val;
	}
}

// Function to print the matrix
void PrintMatrix(vector<int>& a, int n) {
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			cout << a[i * n + j] << " ";
		}
		cout << endl;
	}
}

int main(int argc, char const* argv[]) {
	if (argc != 2) {
		cout << "Usage: ./ver0 <matrix_size> \n\nmatrix_size: Positive Integer" << endl;
		return 1;
	}
	int n = atoi(argv[1]);

	// Size (in bytes) of matrix
	size_t bytes = n * n * sizeof(int);

	// Host vectors
	vector<int> h_a(n * n);
	vector<int> h_b(n * n);
	vector<int> h_c(n * n);

	// Initialize matrices
	Initialize(h_a);
	Initialize(h_b);

	// Allocate device memory
	int* d_a, * d_b, * d_c;
	cudaMalloc(&d_a, bytes);
	cudaMalloc(&d_b, bytes);
	cudaMalloc(&d_c, bytes);

	float gpu_data_transfer_time_ms, gpu_compute_time_ms, gpu_rev_data_transfer_time_ms;

	// some events to count the execution time
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// start to count execution time of GPU version
	cudaEventRecord(start, 0);

	// Copy data to the device
	cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	// compute time elapse on GPU computing
	cudaEventElapsedTime(&gpu_data_transfer_time_ms, start, stop);
	cout << "Host to Device data transfer time: " << gpu_data_transfer_time_ms << "ms" << endl;

	// Blocks per grid dimension (assumes NUM_THREADS_PER_DIM divides N evenly)
	int NUM_BLOCKS_PER_DIM = n / NUM_THREADS_PER_DIM;

	// Use dim3 structs for block  and grid dimensions
	dim3 threadsPerBlock(NUM_THREADS_PER_DIM, NUM_THREADS_PER_DIM);
	dim3 numBlocks(NUM_BLOCKS_PER_DIM, NUM_BLOCKS_PER_DIM);

	cudaEventRecord(start, 0);
	// Launch kernel
	MatrixMul << <numBlocks, threadsPerBlock >> > (d_a, d_b, d_c, n);
	// cudaDeviceSynchronize();
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	// compute time elapse on GPU computing
	cudaEventElapsedTime(&gpu_compute_time_ms, start, stop);
	cout << "GPU compute time: " << gpu_compute_time_ms << "ms" << endl;

	cudaEventRecord(start, 0);
	// Copy back to the host
	cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	// compute time elapse on GPU computing
	cudaEventElapsedTime(&gpu_rev_data_transfer_time_ms, start, stop);
	cout << "Device to Host data transfer time: " << gpu_rev_data_transfer_time_ms << "ms" << endl;

	// Check result on the CPU
#ifdef VALIDATE
	Validate(h_a, h_b, h_c, n);
#endif
	// PrintMatrix(h_c, n);

	// Free memory on device
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);

	return 0;
}
