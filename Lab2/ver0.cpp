#include <bits/stdc++.h>

using namespace std;

// Function implementing SUMMA algorithm
void SUMMA(double** A, double** B, double** C, int n)
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
            matrix[i][j] = (double)rand() / (double)RAND_MAX * 5;
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
            cout << matrix[i][j] << " ";
        }
        cout << endl;
    }
}

int main(int argc, char const* argv[])
{
    if (argc != 3) {
        cout << "Usage: ./ver0 <matrix_size> <num_threads>\n\nmatrix_size: Positive Integer\nnum_threads: Positive Integer" << endl;
        return 1;
    }
    int n = atoi(argv[1]);
    int threads = atoi(argv[2]);
    double** A = new double* [n];
    double** B = new double* [n];
    double** C = new double* [n];

    for (int i = 0; i < n; i++)
    {
        A[i] = new double[n];
        B[i] = new double[n];
        C[i] = new double[n];
    }

    Initialize(A, n);
    Initialize(B, n);
    for (int i = 0; i < n; i++) 
    {
        for (int j = 0; j < n; j++) 
        {
            C[i][j] = 0;
        }
    }

    auto now = chrono::system_clock::now();
    SUMMA(A, B, C, n);
    cout << "millisec=" << std::chrono::duration_cast<std::chrono::milliseconds>(chrono::system_clock::now() - now).count() << "\n";

    return 0;
}