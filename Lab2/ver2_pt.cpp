#include <bits/stdc++.h>

using namespace std;

double** A;
double** B;
double** C;
double** D;
int n;
int threads;

struct thread_data {
    int kk;
    int thread_id;
};

int stride = 50;

// Function implementing SUMMA algorithm using pthreads
void* SUMMA(void* args)
{
    int kk;
    int thread_id;

    struct thread_data* my_data;

    my_data = (struct thread_data*)args;
    thread_id = my_data->thread_id;
    kk = my_data->kk;

    int sqrt_threads = sqrt(threads);
    int chunk_i = thread_id / sqrt_threads;
    int chunk_j = thread_id % sqrt_threads;
    int chunk_size = n / sqrt_threads;
    int max_chunk_i = min((int)(chunk_i + 1) * chunk_size, n);
    int max_chunk_j = min((int)(chunk_j + 1) * chunk_size, n);

    for (int ii = chunk_i * chunk_size; ii < max_chunk_i; ii += stride) 
    {
        for (int jj = chunk_j * chunk_size; jj < max_chunk_j; jj += stride) 
        {
            for (int k = kk; k < kk + stride; ++k)
            {
                for (int i = ii; i < ii + stride; ++i)
                {
                    for (int j = jj; j < jj + stride; ++j)
                    {
                        C[i][j] += A[i][k] * B[k][j];
                    }
                }
            }
        }
    }

    return NULL;
}

// Function implementing SUMMA without any parallelization
void Serial(double** A, double** B, double** C, int n)
{
    for (int k = 0; k < n; k++)
    {
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

// Function to initialize the matrix with random floating point values
void Initialize(double** matrix, int n)
{
    srand((unsigned int)time(NULL));
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            *(*(matrix + i) + j) = ((double)rand() / (double)RAND_MAX) * ((double)RAND_MAX - 1);
        }
    }
}

// Function to print the matrix
void PrintMatrix(double** matrix, int n)
{
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            cout << *(*(matrix + i) + j) << " ";
        }
        cout << endl;
    }
}

// Function to compare the results of parallel and serial version
int Validate(double** A, double** B, int n) 
{
    int mistakes = 0;
    for (int i = 0; i < n; i++) 
    {
        for (int j = 0; j < n; j++) 
        {
            if (A[i][j] != B[i][j]) mistakes++;
        }
    }
    return mistakes;
}

int main(int argc, char const* argv[])
{
    if (argc != 3) {
        cout << "Usage: ./ver0 <matrix_size> <num_threads>\n\nmatrix_size: Positive Integer\nnum_threads: Positive Integer" << endl;
        return 1;
    }
    n = atoi(argv[1]);
    threads = atoi(argv[2]);

    A = (double**)malloc(sizeof(double*) * n);
    B = (double**)malloc(sizeof(double*) * n);
    C = (double**)malloc(sizeof(double*) * n);
    D = (double**)malloc(sizeof(double*) * n);

    for (int i = 0; i < n; i++)
    {
        A[i] = (double*)malloc(sizeof(double) * n);
        B[i] = (double*)malloc(sizeof(double) * n);
        C[i] = (double*)malloc(sizeof(double) * n);
        D[i] = (double*)malloc(sizeof(double) * n);
    }

    for (int i = 0; i < n; i++) 
    {
        for (int j = 0; j < n; j++) 
        {
            *(*(C + i) + j) = 0;
        }
    }

    for (int i = 0; i < n; i++) 
    {
        for (int j = 0; j < n; j++) 
        {
            *(*(D + i) + j) = 0;
        }
    }

    pthread_t workers[threads];
    struct thread_data thread_data_array[threads];

    auto now = chrono::system_clock::now();
    for (long kk = 0; kk < n; kk += stride) 
    {
        for (long thread_id = 0; thread_id < threads; thread_id++) 
        {
            thread_data_array[thread_id].kk = kk;
            thread_data_array[thread_id].thread_id = thread_id;
            pthread_create(workers + thread_id, NULL, SUMMA, (void*)&thread_data_array[thread_id]);
        }

        for (int thread_id = 0; thread_id < threads; thread_id++)
            pthread_join(workers[thread_id], NULL);
    }
    cout << "millisec=" << std::chrono::duration_cast<std::chrono::milliseconds>(chrono::system_clock::now() - now).count() << "\n";

    now = chrono::system_clock::now();
    Serial(A, B, D, n);
    cout << "millisec=" << std::chrono::duration_cast<std::chrono::milliseconds>(chrono::system_clock::now() - now).count() << "\n";

    cout << Validate(C, D, n) << endl;

    return 0;
}