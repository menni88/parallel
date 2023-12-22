#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define SIZE 8

int getOwner(int vertex  ,int MPI_COMM_SIZE) {
    return vertex % MPI_COMM_SIZE;
}

void bfs(int localGraph[SIZE][SIZE], int localSource, int localVisited[], int localVisitedCount , int rank , int size ) {
    
    int level = 0;
    int frontier[SIZE];
    int frontierSize = 1;
    frontier[0] = localSource;
    
    while (frontierSize > 0) {
        int nextFrontier[SIZE];
        int nextFrontierSize = 0;

        for (int i = 0; i < frontierSize; i++) {
            int currentVertex = frontier[i];
            localVisited[localVisitedCount++] = currentVertex;

            // Process neighbors and add them to the next frontier
            for (int j = 0; j < SIZE; j++) {
                if (localGraph[currentVertex][j] == 1) {
                    int owner = getOwner(j + 1 , size);
                    if (owner == rank) {
                        // Local vertex, add to next frontier
                        nextFrontier[nextFrontierSize++] = j;
                    } else {
                        // Non-local vertex, send to owner process
                        MPI_Send(&j, 1, MPI_INT, owner, 0, MPI_COMM_WORLD);
                    }
                }
            }
        }

        // Use MPI_Alltoall to exchange the next frontier among all processes
        MPI_Alltoall(nextFrontier, 1, MPI_INT, frontier, 1, MPI_INT, MPI_COMM_WORLD);
        frontierSize = nextFrontierSize;

        // Increment the level for synchronization
        level++;
        MPI_Barrier(MPI_COMM_WORLD);
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int localGraph[SIZE / size][SIZE]; // Assuming equal division of the graph
    int graphData[SIZE][SIZE] = {
        {0, 1, 0, 1, 0, 0, 0, 0},
        {1, 0, 1, 0, 0, 0, 0, 0},
        {0, 1, 0, 1, 1, 0, 0, 0},
        {1, 0, 1, 0, 0, 1, 0, 0},
        {0, 0, 1, 0, 0, 0, 1, 0},
        {0, 0, 0, 1, 0, 0, 0, 1},
        {0, 0, 0, 0, 1, 0, 0, 0},
        {0, 0, 0, 0, 0, 1, 0, 0}
    };
    
    int localVisited[SIZE];
    int localVisitedCount = 0;

    MPI_Scatter(graphData, SIZE / size * SIZE, MPI_INT, localGraph, SIZE / size * SIZE, MPI_INT, 0, MPI_COMM_WORLD);
    int localSource=-1;
    if(rank==0){
      int globalSource = 0;
      localSource=(globalSource-rank*(SIZE / size)+SIZE)%SIZE;
    }
    MPI_Bcast(&localSource, 1, MPI_INT, 0, MPI_COMM_WORLD);

    bfs(localGraph, localSource, localVisited, localVisitedCount, rank , size);

    if (rank == 0) {

        int globalVisited[SIZE];
        MPI_Gather(localVisited, SIZE, MPI_INT, globalVisited, SIZE, MPI_INT, 0, MPI_COMM_WORLD);

        printf("Visited Order: ");
        for (int i = 0; i < SIZE; i++) {
            printf("%d ", globalVisited[i]);
        }
        printf("\n");
    }

    MPI_Finalize();
    return 0;
}
