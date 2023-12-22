#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <mpi.h>
#include <math.h>

#define MAX_VERTICES 100
#define V graph[0][0]

int find_owner(int vertex, int size)
{
    return ceil((float)vertex / size) - 1;
}

void distributed_BFS(int graph[2][MAX_VERTICES], int source, int rank, int size, MPI_Comm comm, int distance[MAX_VERTICES])
{
    int E[2][MAX_VERTICES];

    for (int i = 0; i < 2; i++)
    {
        for (int j = 0; j < MAX_VERTICES; j++)
        {
            E[i][j] = graph[i][j];
        }
    }

    // Create a boolean array to track visited vertices
    bool visited[V];
    for (int i = 0; i < V; i++)
    {
        visited[i] = false;
    }

    // Broadcast the source vertex to all processes
    MPI_Bcast(&source, 1, MPI_INT, 0, comm);

    // Initialize BFS queue
    int queue[V];
    int front = -1, rear = -1;

    // Add the source vertex to the queue
    queue[++rear] = source;

    // Calculate the send counts and displacements for scatterv
    int sendcounts[size];
    int displs[size];
    int rem = V % size;
    int sum = 0;
    for (int i = 0; i < size; i++)
    {
        sendcounts[i] = (V / size) * V;
        if (rem > 0)
        {
            sendcounts[i] += V;
            rem--;
        }
        displs[i] = sum;
        sum += sendcounts[i];
    }

    // Allocate memory for the local submatrix
    int local_V = sendcounts[rank] / V + 1; // Adding 1 for safety
    int local_E[local_V][V];

    // Scatter the submatrices to all processes
    MPI_Scatterv(E, sendcounts, displs, MPI_INT, local_E, local_V * V, MPI_INT, 0, comm);

    // Scatter the distance array to all processes
    int local_distance[local_V];

    MPI_Scatter(distance, local_V, MPI_INT, local_distance, local_V, MPI_INT, 0, comm);

    while (front != rear)
    {
        int u = queue[++front]; // Declare and initialize u

        for (int v = 0; v < V; v++)
        {
            if (local_E[u % local_V][v] == 1 && !visited[v])
            {
                if (find_owner(v, size) == rank)
                {
                    local_distance[v] = local_distance[u] + 1;
                    visited[v] = true;
                    queue[++rear] = v;
                }
            }
        }
    }

    MPI_Barrier(comm);

    while (front != rear)
    {
        int u = queue[++front];
        // Calculate the row index for the submatrix
        int row = u - displs[rank] / V;

        // Check the adjacent vertices in the submatrix
        for (int v = 0; v < V; v++)
        {
            if (local_E[row][v] == 1 && !visited[v])
            {
                if (find_owner(v, size) == rank)
                {
                    local_distance[v] = local_distance[u] + 1;
                    visited[v] = true;
                    queue[++rear] = v;
                }
            }
        }
    }

    MPI_Barrier(comm);

    // Gather the distance array from all processes
    MPI_Gather(local_distance, local_V, MPI_INT, distance, local_V, MPI_INT, 0, comm);
}

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int graph[2][MAX_VERTICES] = {{6}, {0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0}};
    int source_vertex = 0;
    int distance[MAX_VERTICES];

    // Initialize distances to -1
    for (int i = 0; i < V; i++)
    {
        distance[i] = -1;
    }

    // Set the distance of the source vertex to 0
    distance[source_vertex] = 0;

    // Calculate the send counts and displacements for scatterv
    int sendcounts[size];
    int displs[size];
    int rem = V % size;
    int sum = 0;
    for (int i = 0; i < size; i++)
    {
        sendcounts[i] = (V / size) * V;
        if (rem > 0)
        {
            sendcounts[i] += V;
            rem--;
        }
        displs[i] = sum;
        sum += sendcounts[i];
    }

    int local_V = sendcounts[rank] / V + 1; // Adding 1 for safety

    // Scatter the distance array to all processes
    int local_distance[local_V];

    MPI_Scatter(distance, local_V, MPI_INT, local_distance, local_V, MPI_INT, 0, MPI_COMM_WORLD);

    distributed_BFS(graph, source_vertex, rank, size, MPI_COMM_WORLD, local_distance);

    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0)
    {
        printf("Rank 0: Result = [");
        for (int i = 0; i < V; i++)
        {
            printf("%d ", distance[i]);
        }
        printf("]\n");
    }

    MPI_Finalize();

    return 0;
}

