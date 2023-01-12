#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <vector>
#include <chrono>

#define TILE_SIZE 32

using namespace std;

__global__ void MatrixMul(const int* a, const int* b, int* c, clock_t* mem_time, clock_t* compute_time, int n) {
    
    int t_x = threadIdx.x;
    int t_y = threadIdx.y;
    int b_x = blockIdx.x;
    int b_y = blockIdx.y;

    // Compute each thread's global row and column index
    int row = b_y * blockDim.y + t_y;
    int col = b_x * blockDim.x + t_x;

    // Shared memory between threads in a block
    __shared__ int s_a[TILE_SIZE * TILE_SIZE];
    __shared__ int s_b[TILE_SIZE * TILE_SIZE];

    // Accumulate in temporary variable
    int tmp = 0;

    clock_t mem_time_per_block = 0;
    clock_t compute_time_per_block = 0;

    // Iterate over each tile
    for (int k = 0; k < (n / TILE_SIZE); ++k) {
        clock_t now = clock();

        // get the corresponding tile ids of matrix A and matrix B
        int tile_A_i = TILE_SIZE * b_y;
        int tile_A_j = TILE_SIZE * k;
        int tile_B_i = TILE_SIZE * k;
        int tile_B_j = TILE_SIZE * b_x;

        // Load elements of this tile into the shared memory
        s_a[t_y * TILE_SIZE + t_x] = a[(t_x + tile_A_j) + (t_y + tile_A_i) * n];
        s_b[t_y * TILE_SIZE + t_x] = b[(t_x + tile_B_j) + (t_y + tile_B_i) * n];

        // Wait for all threads to load data into shared memory
        __syncthreads();

        mem_time_per_block += (clock() - now);

        now = clock();
        // Perform matrix multiplication for the given tile
        for (int i = 0; i < TILE_SIZE; ++i) {
            tmp += s_a[i + TILE_SIZE * t_y] * s_b[t_x + TILE_SIZE * i];
        }

        // Wait for all threads to finish using the current tile before loading a new tile
        __syncthreads();

        compute_time_per_block += (clock() - now);
    }

    // Write back results
    c[row * n + col] = tmp;

    mem_time[blockIdx.y * TILE_SIZE + blockIdx.x] = mem_time_per_block;
    compute_time[blockIdx.y * TILE_SIZE + blockIdx.x] = compute_time_per_block;
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
                cout << "Mismatch " << tmp << " != " << c[i * n + j] << " at (" << i << ", " << j << ")" << endl;
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

// Function to calculate average time required per block
// Using incremental average calculation method to avoid overflow
float AvgTime(vector<clock_t>& time_stats, int clock_rate) {
    float avg_time = 0;
    for (auto i = time_stats.begin(); i < time_stats.end(); i++) {
        // float idx = 1/(i - time_stats.begin() + 1);
        // // cout << *i << endl;
        // avg_time += idx*((float) *i/clock_rate - avg_time);
        avg_time += (float)*i / clock_rate;
    }
    avg_time /= time_stats.size();

    return avg_time;
}

int main(int argc, char const* argv[]) {
    if (argc != 2) {
        cout << "Usage: ./ver0 <matrix_size> \n\nmatrix_size: Positive Integer" << endl;
        return 1;
    }
    int n = atoi(argv[1]);

    int clock_rate = 0;
    int device = 0;
    cudaDeviceGetAttribute(&clock_rate, cudaDevAttrClockRate, device);

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

    // Blocks per grid dimension (assumes TILE_SIZE divides N evenly)
    int NUM_BLOCKS_PER_DIM = n / TILE_SIZE;
    // int shmem_size = NUM_BLOCKS_PER_DIM * NUM_BLOCKS_PER_DIM;

    // Use dim3 structs for block  and grid dimensions
    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 numBlocks(NUM_BLOCKS_PER_DIM, NUM_BLOCKS_PER_DIM);

    size_t time_bytes = NUM_BLOCKS_PER_DIM * NUM_BLOCKS_PER_DIM * sizeof(clock_t);

    // Allocate device memory for time calculation
    clock_t* mem_time, * compute_time;
    cudaMalloc(&mem_time, time_bytes);
    cudaMalloc(&compute_time, time_bytes);

    cudaEventRecord(start, 0);
    // Launch kernel
    MatrixMul << <numBlocks, threadsPerBlock >> > (d_a, d_b, d_c, mem_time, compute_time, n);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    // compute time elapse on GPU computing
    cudaEventElapsedTime(&gpu_compute_time_ms, start, stop);
    cout << "GPU compute time: " << gpu_compute_time_ms << "ms" << endl;

    // Allocate host memory for time calculation
    vector<clock_t> host_mem_time(NUM_BLOCKS_PER_DIM * NUM_BLOCKS_PER_DIM);
    vector<clock_t> host_compute_time(NUM_BLOCKS_PER_DIM * NUM_BLOCKS_PER_DIM);

    // Copy time statistics back to host
    cudaMemcpy(host_mem_time.data(), mem_time, time_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(host_compute_time.data(), compute_time, time_bytes, cudaMemcpyDeviceToHost);

    cout << "Average time for memory related operations per block: " << AvgTime(host_mem_time, clock_rate) << "ms" << endl;
    cout << "Average time for compute related operations per block: " << AvgTime(host_compute_time, clock_rate) << "ms" << endl;

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

    // Free memory on host
    vector<int>().swap(h_a);
    vector<int>().swap(h_b);
    vector<int>().swap(h_c);

    return 0;
}