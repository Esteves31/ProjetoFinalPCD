#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

#define N 2000       // Tamanho da grade
#define T 500        // Número de iterações no tempo
#define D 0.1        // Coeficiente de difusão
#define DELTA_T 0.01
#define DELTA_X 1.0

// Função para alocar uma matriz 2D contínua
double** allocate_matrix(int rows, int cols) {
    double* data = (double*)malloc(rows * cols * sizeof(double));
    if (data == NULL) {
        return NULL;
    }
    double** array = (double**)malloc(rows * sizeof(double*));
    if (array == NULL) {
        free(data);
        return NULL;
    }
    for (int i = 0; i < rows; i++) {
        array[i] = &(data[i * cols]);
    }
    return array;
}

// Função para liberar a matriz alocada
void free_matrix(double** matrix) {
    if (matrix != NULL) {
        free(matrix[0]); 
        free(matrix);    
    }
}

// Função principal para a equação de difusão
void diff_eq(double** C, double** C_new, int local_rows, int n, int rank, int size) {
    for (int t = 0; t < T; t++) {

        MPI_Status status;
        if (rank != 0) {
            MPI_Sendrecv(C[1], n, MPI_DOUBLE, rank - 1, 0,
                         C[0], n, MPI_DOUBLE, rank - 1, 1,
                         MPI_COMM_WORLD, &status);
        } else {
            for (int j = 0; j < n; j++) {
                C[0][j] = 0.0;
            }
        }

        if (rank != size - 1) {
            MPI_Sendrecv(C[local_rows], n, MPI_DOUBLE, rank + 1, 1,
                         C[local_rows + 1], n, MPI_DOUBLE, rank + 1, 0,
                         MPI_COMM_WORLD, &status);
        } else {
            for (int j = 0; j < n; j++) {
                C[local_rows + 1][j] = 0.0;
            }
        }

        for (int i = 1; i <= local_rows; i++) {
            for (int j = 1; j < n - 1; j++) {
                C_new[i][j] = C[i][j] + D * DELTA_T * (
                    (C[i+1][j] + C[i-1][j] + C[i][j+1] + C[i][j-1] - 4.0 * C[i][j]) / (DELTA_X * DELTA_X)
                );
            }
        }

        double difmedio_local = 0.0;
        for (int i = 1; i <= local_rows; i++) {
            for (int j = 1; j < n - 1; j++) {
                difmedio_local += fabs(C_new[i][j] - C[i][j]);
                C[i][j] = C_new[i][j];
            }
        }

        double difmedio_global = 0.0;
        MPI_Reduce(&difmedio_local, &difmedio_global, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

        if ((t % 100) == 0 && rank == 0) {
            double media = difmedio_global / ((n - 2) * (n - 2));
            printf("Iteracao %d - diferenca=%.6g\n", t, media);
        }
    }
}

int main(int argc, char *argv[]) {
    int rank, size;
    double **C = NULL, **C_new = NULL;
    int local_rows;
    int remainder;
    double start_time, end_time;
    int global_center = N / 2;

    MPI_Init(&argc, &argv);                   
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);     
    MPI_Comm_size(MPI_COMM_WORLD, &size);     

    if (size > N) {
        if (rank == 0) {
            fprintf(stderr, "Número de processos (%d) excede o número de linhas (%d).\n", size, N);
        }
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    local_rows = N / size;
    remainder = N % size;
    if (rank < remainder) {
        local_rows += 1;
    }

    C = allocate_matrix(local_rows + 2, N);
    C_new = allocate_matrix(local_rows + 2, N);
    if (C == NULL || C_new == NULL) {
        fprintf(stderr, "Processo %d: Falha na alocacao de memoria\n", rank);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    for (int i = 0; i < local_rows + 2; i++) {
        for (int j = 0; j < N; j++) {
            C[i][j] = 0.0;
            C_new[i][j] = 0.0;
        }
    }

    int start_row = rank * (N / size) + (rank < remainder ? rank : remainder);
    int end_row = start_row + local_rows - 1;

    if (global_center >= start_row && global_center <= end_row) {
        int local_i = global_center - start_row + 1; 
        C[local_i][N / 2] = 1.0;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime();

    diff_eq(C, C_new, local_rows, N, rank, size);

    MPI_Barrier(MPI_COMM_WORLD); 
    end_time = MPI_Wtime();

    double final_conc = 0.0;
    if (global_center >= start_row && global_center <= end_row) {
        int local_i = global_center - start_row + 1;
        final_conc = C[local_i][N / 2];
    }

    double global_final_conc = 0.0;
    MPI_Reduce(&final_conc, &global_final_conc, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    double exec_time = end_time - start_time;
    double max_exec_time;
    MPI_Reduce(&exec_time, &max_exec_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Concentracao final no centro: %f\n", global_final_conc);
        printf("Tempo de execucao (MPI): %f segundos\n", max_exec_time);
    }

    free_matrix(C);
    free_matrix(C_new);

    MPI_Finalize(); 
    return 0;
}