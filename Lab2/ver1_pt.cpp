// #include <pthread.h>
#include <bits/stdc++.h>
#include <unistd.h>

using namespace std;

double **A;
double **B;
double **C;
double **D;
int n;
int threads;
int sqrt_threads;
int chunk_size;

struct thread_data {
    int k;
    int thread_id;
};

// Function implementing SUMMA algorithm
void* SUMMA(void *args)
{       
    int k;
    int thread_id;

    // struct thread_data *my_data;

    thread_data *my_data = (thread_data *) args;

    thread_id = my_data->thread_id;
    k = my_data->k;

    int chunk_i = thread_id/sqrt_threads;
    int chunk_j = thread_id%sqrt_threads;
    int max_chunk_i = min((int)(chunk_i+1)*chunk_size, n);
    int max_chunk_j = min((int)(chunk_j+1)*chunk_size, n);

    for (int i=chunk_i*chunk_size; i< max_chunk_i; ++i) {
        for (int j=chunk_j*chunk_size; j< max_chunk_j; ++j) {
            C[i][j] += A[i][k]*B[k][j];
        }
    }

    pthread_exit(0);
}

void Serial(double** A, double** B, double** C, int n)
{
    for (int k = 0; k < n; k++)
    {
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                C[i][j] += A[i][k]*B[k][j];
            }
        }
    }
}

// Function to initialize the matrix with random floating point values
void Initialize(double** matrix, int n)
{
    srand((unsigned int)time(NULL));
    sleep(1);
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            *(*(matrix+i)+j) = ((double)rand() / (double)RAND_MAX)*((double)RAND_MAX - 1);
        }
    }
}

//Function to print the matrix
void PrintMatrix(double** matrix, int n)
{
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            cout << *(*(matrix+i)+j) << " ";
        }
        cout << endl;
    }
}

int Validate(double** A, double** B, int n) {
    int mistakes = 0;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (A[i][j] != B[i][j]) mistakes++;
        }
    }
    return mistakes;
}

int main(int argc, char const* argv[])
{
    if (argc != 3) {
        cout << "Usage: ./ver0 <matrix_size> <threads>" << endl;
        return 1;
    }
    n = atoi(argv[1]);
    threads = atoi(argv[2]);
    sqrt_threads = sqrt(threads);
    chunk_size = n/sqrt_threads;

    A = (double **)malloc(sizeof(double*) * n);
    B = (double **)malloc(sizeof(double*) * n);
    C = (double **)malloc(sizeof(double*) * n);
    D = (double **)malloc(sizeof(double*) * n);

    for (int i = 0; i < n; i++)
    {
        A[i]= (double*)malloc(sizeof(double) * n);
        B[i]= (double*)malloc(sizeof(double) * n);
        C[i]= (double*)malloc(sizeof(double) * n);
        D[i]= (double*)malloc(sizeof(double) * n);
    }

    Initialize(A, n);
    Initialize(B, n);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            *(*(C+i)+j) = 0;
        }
    }

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            *(*(D+i)+j) = 0;
        }
    }

    // cout << "Matrix A:" << endl;
    // PrintMatrix(A, n);
    // cout << "Matrix B:" << endl;
    // PrintMatrix(B, n);

    pthread_t workers[threads];

    struct thread_data thread_data_array[threads];

    auto now = chrono::system_clock::now();
    for (long k = 0; k < n; ++k) {
        for (long thread_id = 0; thread_id < threads; ++thread_id) {
            thread_data_array[thread_id].k = k;
            thread_data_array[thread_id].thread_id = thread_id;
            pthread_create(workers + thread_id, NULL, SUMMA, (void *) &thread_data_array[thread_id]);
        }

        for (int thread_id = 0; thread_id < threads; ++thread_id)
            pthread_join(workers[thread_id], NULL);
    }
    cout << "millisec=" << std::chrono::duration_cast<std::chrono::milliseconds>(chrono::system_clock::now() - now).count() << "\n";

    now = chrono::system_clock::now();
    Serial(A, B, D, n);
    cout << "millisec=" << std::chrono::duration_cast<std::chrono::milliseconds>(chrono::system_clock::now() - now).count() << "\n";

    cout << Validate(C, D, n) << endl;

    return 0;
}