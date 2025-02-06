#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <omp.h>

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
        free(matrix[0]); // Liberar os dados contínuos
        free(matrix);    // Liberar os apontadores de linha
    }
}

// Função principal para a equação de difusão (versão híbrida MPI + OpenMP)
void diff_eq_hybrid(double** C, double** C_new, int local_rows, int n, int rank, int size) {
    for (int t = 0; t < T; t++) {

        // Comunicação de halo: enviar e receber linhas para os processos vizinhos
        MPI_Status status;
        // Enviar a primeira linha de dados para cima e receber a última linha do vizinho de cima
        if (rank != 0) {
            MPI_Sendrecv(C[1], n, MPI_DOUBLE, rank - 1, 0,
                        C[0], n, MPI_DOUBLE, rank - 1, 1,
                        MPI_COMM_WORLD, &status);
        } else {
            // Se for o primeiro processo, definir a linha de halo superior como zero
            #pragma omp parallel for
            for (int j = 0; j < n; j++) {
                C[0][j] = 0.0;
            }
        }

        // Enviar a última linha de dados para baixo e receber a primeira linha do vizinho de baixo
        if (rank != size - 1) {
            MPI_Sendrecv(C[local_rows], n, MPI_DOUBLE, rank + 1, 1,
                        C[local_rows + 1], n, MPI_DOUBLE, rank + 1, 0,
                        MPI_COMM_WORLD, &status);
        } else {
            // Se for o último processo, definir a linha de halo inferior como zero
            #pragma omp parallel for
            for (int j = 0; j < n; j++) {
                C[local_rows + 1][j] = 0.0;
            }
        }

        // Calcular C_new com base em C utilizando OpenMP para paralelização
        #pragma omp parallel for collapse(2)
        for (int i = 1; i <= local_rows; i++) {
            for (int j = 1; j < n - 1; j++) {
                C_new[i][j] = C[i][j] + D * DELTA_T * (
                    (C[i+1][j] + C[i-1][j] + C[i][j+1] + C[i][j-1] - 4.0 * C[i][j]) / (DELTA_X * DELTA_X)
                );
            }
        }

        // Calcular a diferença média e atualizar C com OpenMP para paralelização
        double difmedio_local = 0.0;

        #pragma omp parallel for reduction(+:difmedio_local) collapse(2)
        for (int i = 1; i <= local_rows; i++) {
            for (int j = 1; j < n - 1; j++) {
                difmedio_local += fabs(C_new[i][j] - C[i][j]);
                C[i][j] = C_new[i][j];
            }
        }

        // Reduzir a difmedio_local para difmedio_global no processo 0
        double difmedio_global = 0.0;
        MPI_Reduce(&difmedio_local, &difmedio_global, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

        // Processo 0 imprime a diferença a cada 100 iterações
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

    // Inicializar o ambiente MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);     // Obtém o rank do processo
    MPI_Comm_size(MPI_COMM_WORLD, &size);     // Obtém o número de processos

    // Validação: garantir que o número de processos não exceda o número de linhas
    if (size > N) {
        if (rank == 0) {
            fprintf(stderr, "Número de processos (%d) excede o número de linhas (%d).\n", size, N);
        }
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    // Dividir as linhas entre os processos
    local_rows = N / size;
    remainder = N % size;
    if (rank < remainder) {
        local_rows += 1;
    }

    // Alocar as matrizes C e C_new com linhas extras para halo
    C = allocate_matrix(local_rows + 2, N);
    C_new = allocate_matrix(local_rows + 2, N);
    if (C == NULL || C_new == NULL) {
        fprintf(stderr, "Processo %d: Falha na alocacao de memoria\n", rank);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    // Inicializar as matrizes com 0
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < local_rows + 2; i++) {
        for (int j = 0; j < N; j++) {
            C[i][j] = 0.0;
            C_new[i][j] = 0.0;
        }
    }

    // Determinar quais linhas cada processo possui
    int start_row = rank * (N / size) + (rank < remainder ? rank : remainder);
    int end_row = start_row + local_rows - 1;

    // Inicializar uma concentração alta no centro
    if (global_center >= start_row && global_center <= end_row) {
        int local_i = global_center - start_row + 1; // +1 devido ao halo superior
        C[local_i][N / 2] = 1.0;
    }

    // Configurar o número de threads OpenMP
    int num_threads = 2; // Defina de acordo com seu ambiente
    omp_set_num_threads(num_threads);

    // Iniciar temporizador
    MPI_Barrier(MPI_COMM_WORLD); // Sincronizar todos os processos
    start_time = MPI_Wtime();

    // Executar as iterações no tempo para a equação de difusão (híbrido MPI + OpenMP)
    diff_eq_hybrid(C, C_new, local_rows, N, rank, size);

    // Finalizar temporizador
    MPI_Barrier(MPI_COMM_WORLD); // Sincronizar todos os processos
    end_time = MPI_Wtime();

    // Obter a concentração final no centro
    double final_conc = 0.0;
    if (global_center >= start_row && global_center <= end_row) {
        int local_i = global_center - start_row + 1;
        final_conc = C[local_i][N / 2];
    }

    // Reduzir para obter a concentração final no processo 0
    double global_final_conc = 0.0;
    MPI_Reduce(&final_conc, &global_final_conc, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    // Obter o tempo de execução total
    double exec_time = end_time - start_time;
    double max_exec_time;
    MPI_Reduce(&exec_time, &max_exec_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // Processo 0 imprime os resultados finais
    if (rank == 0) {
        printf("Concentracao final no centro: %f\n", global_final_conc);
        printf("Tempo de execucao (MPI + OpenMP): %f segundos\n", max_exec_time);
    }

    // Liberar a memória alocada
    free_matrix(C);
    free_matrix(C_new);

    // Finalizar o ambiente MPI
    MPI_Finalize();
    return 0;
}