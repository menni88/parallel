#include <iostream>
#include <fstream>
#include <mpi.h>
#include <stdlib.h>
#include <cstring>
#include <vector>
#include <queue>
#include <vector>

// реализован последовательный обход в ширину (bfs), а также параллельный с использованием технологии mpi.
// для демонстрации практически не имеет смысла менять исходную вершину, поэтому она "захардкожена" значением 0.
// в основу распараллеливание заложена идея о том, что параллельно обрабатывается каждый уровень.
// затем идёт распределение между всеми рангами в комммуникаторе соответствующих вершин "фронтира", т.е.
// находится "хозяин" каждой вершины и именно он на следующем уровне будет обрабатывать конкретную вершину.
// завершение параллельной обработки происходит при пустой очереди у всех рангов, реализовано через All_Reduce.
// в программе реализовано сравнение времени вычисления последовательным и параллельным подходами.
// при запуске требуется ввести количество вершин (большее, чем количество запущенных ветвей), а также указать,
// необходимо ли сохранить матрицу и векторы расстояний.

// последовательная обработка
void serial(int n, int* adjacency_matrix, int save)
{
    std::queue<int> q; // очередь для bfs
    q.push(0); 
    std::vector<bool> used(n); // уже пройденные вершины
    std::vector<int> d(n); // расстояние до вершины
    used[0] = true; 

    // пока не пуста очередь - просматриваем вершины
    while (!q.empty())
    {
        // просматриваем ближайшую из очереди вершину и убираем её из очереди
        int v = q.front();
        q.pop();

        // просматриваем всех соседей текущей вершины
        for (int i = 0; i < n; i++)
        {
            // если есть ребро и не посещена ранее
            // то заносим в очередь, отмечаем посещённой и вычисляем расстояние
            int to = adjacency_matrix[v * n + i]; 
            if (to == 1 && !used[i])
            {
                used[i] = true;
                q.push(i);
                d[i] = d[v] + 1;
            }
        }
    }

    if (save == 1)
    {
        std::cout << "Saving distance vector to file \"distance_vector.txt\"..." << std::endl << std::flush;
        std::ofstream path_file("distance_vector.txt");
        path_file << "Distance vector, serial, size = " << n << "\n";
        for (int i = 0; i < n; i++)
        {
            path_file << d[i] << " ";
        }
        path_file << "\n\n";
    }
}

// нахождение "хозяина" вершины
int find_owner(int n, int size, int val)
{
    int owner = 0;
    int count = 0;
    int distance = n / size;
    while (count + distance <= val && owner != size - 1)
    {
        owner++;
        count += distance;
    }

    return owner;
}

// вспомогательный метод для коррекции значения вершины внутри ранга
int adjust_vertex(int n, int size, int val)
{
    return val - find_owner(n, size, val) * (n / size);
}

// параллельная обработка
void parallel(int n, int* adjacency_matrix, int rank, int size, int save)
{
    int level = 0; // текущий уровень
    bool alive = true; // флаг продолжения вычисления
    std::queue<int> fs, ns; // "фронтир"-очередь и очередь для следующего уровня
    std::vector<bool> used(n); // вектор пройденных вершин. локальный, используется для небольшой оптимизации
    std::vector<int> d(n); // расстояние до вершин
    int* sendcounts = (int*)malloc(sizeof(int) * size); // количество вершин для каждого из рангов
    int* displs = (int*)malloc(sizeof(int) * size); // смещение в матрице смежности для каждого из рангов
    
    // вычисление количества вершин и смещения в матрице смежности
    int count = n;
    for (int i = 0; i < size - 1; i++)
    {
        sendcounts[i] = (n / size) * n;
        displs[i] = (n - count) * n;
        count -= (n / size);
    }
    sendcounts[size - 1] = count * n;
    displs[size - 1] = (n - count) * n;

    // распределение матрицы смежности для каждого ранга
    int* adjacency_thread = (int*)malloc(sizeof(int) * n * n);
    MPI_Scatterv(adjacency_matrix, sendcounts, displs, MPI_INT, adjacency_thread, n * n, MPI_INT, 0, MPI_COMM_WORLD);

    // инициализация обработки происходит корневым (нулевым) рангом
    if (rank == 0)
    {
        fs.push(0);
        used[0] = true;
        d[0] = level;
    }

    // обработка идёт, пока "фронтир"-очередь хотя бы у одного из рангов непуста
    while (alive)
    {
        level++;
        // пока очередь на текущем уровне не пуста, просматриваем вершины
        while (!fs.empty())
        {
            // просматриваем ближайшую из "фронтир"-очереди вершину и убираем её из очереди
            int v = fs.front();
            fs.pop();
            // просматриваем всех соседей текущей вершины
            for (int i = 0; i < n; i++)
            {
                // если есть ребро и не посещена ранее
                // то заносим в очередь следующего уровня и отмечаем посещённой
                int to = adjacency_thread[adjust_vertex(n, size, v) * n + i];
                if (to == 1 && !used[i])
                {
                    used[i] = true;
                    ns.push(i);
                }
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);

        // формирование каждым рангом массива с очередью следующего уровня
        bool* send_q = (bool*)calloc(n, sizeof(bool));
        while (!ns.empty())
        {
            int val = ns.front();
            ns.pop();
            send_q[val] = true;
            d[val] = level;
        }

        // если ранг - корневой, то принимаем очереди от остальных рангов, иначе - отправляем
        if (rank == 0)
        {
            // инициализация массива для обработки всех рангов
            bool* recv_q = (bool*)calloc(n, sizeof(bool));
            memcpy(recv_q, send_q, sizeof(bool) * n);

            // обработка всех некорневых рангов
            for (int i = 1; i < size; i++)
            {
                MPI_Recv(send_q, n, MPI_C_BOOL, i, i, MPI_COMM_WORLD, MPI_STATUS_IGNORE); // приём очереди в массив send_q
                for (int j = 0; j < n; j++)
                {
                    // если вершина содержится в очереди, то устанавливаем соответствующий флаг в массиве recv_q, вычисляем расстояние
                    if (send_q[j] == true)
                    {
                        recv_q[j] = true;
                        used[j] = true;
                        if (d[j] == 0 && j != 0) d[j] = level;
                    }
                }
            }

            // нулевой ранг заносит в свою "фронтир"-очередь "принадлежащие" ему вершины
            for (int i = 0; i < n / size; i++) if (recv_q[i]) fs.push(i);
            // рассылка массива с очередями следующего уровня всем остальным рангам
            for (int i = 1; i < size; i++) MPI_Send(recv_q, n, MPI_C_BOOL, i, 0, MPI_COMM_WORLD);
            free(recv_q);

        }
        else
        {
            MPI_Send(send_q, n, MPI_C_BOOL, 0, rank, MPI_COMM_WORLD); // отправка очереди следующего уровня 0 рангу
            MPI_Recv(send_q, n, MPI_C_BOOL, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE); // приём общей очереди следующего уровня
            // занесение в свою "фронтир"-очередь "принадлежащие" текущему рангу вершин
            if (rank != size - 1)
            {
                for (int i = (n / size) * rank; i < (n / size) * (rank + 1); i++) if (send_q[i]) fs.push(i);
            }
            else for (int i = (n / size) * rank; i < n; i++) if (send_q[i]) fs.push(i);
        }
        MPI_Barrier(MPI_COMM_WORLD);

        // вычисление и рассылка флага продолжения вычисления
        bool send_alive = fs.empty() ? false : true;
        MPI_Allreduce(&send_alive, &alive, 1, MPI_C_BOOL, MPI_LOR, MPI_COMM_WORLD);

        free(send_q);
    }

    if (rank == 0)
    {
        if (save == 1)
        {
            std::cout << "Saving distance vector to file \"distance_vector.txt\"..." << std::endl << std::flush;
            std::ofstream path_file("distance_vector.txt", std::ios_base::app);
            path_file << "Distance vector, parallel, size = " << n << "\n";
            for (int i = 0; i < n; i++)
            {
                path_file << d[i] << " ";
            }
            path_file << "\n";
        }
    }

    free(adjacency_thread);
    free(sendcounts);
    free(displs);
}

int main(int argc, char* argv[])
{
    // инициализация MPI
    srand(time(NULL));
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n, save;
	int* adjacency_matrix = NULL;
	double start;
	setlocale(LC_ALL, "russian");

    // корневой ранг считывает входные данные
	if (rank == 0)
	{
		std::cout << "Enter N - adjecency matrix size (int > 0): ";
		std::cin >> n;
        std::cout << "Enter 1 to save further results: matrix and distance vector (could take long time and big size) or 0 to skip: ";
        std::cin >> save;
	}

    // рассылка входных данных всем потокам
	MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&save, 1, MPI_INT, 0, MPI_COMM_WORLD);

	if (n == 0)
	{
		if (rank == 0) std::cout << "N must be bigger than 0" << std::endl;
		MPI_Finalize();
		return 0;
	}

    if (size > n)
    {
        if (rank == 0) std::cout << "Please startup program with param -n less or equal to entered N" << std::endl;
        MPI_Finalize();
        return 0;
    }

    // корневой ранг генерирует матрицу смежности и производит последовательную обработку
	if (rank == 0)
	{
		std::cout << "Generating adjacency matrix..." << std::endl << std::flush;
		adjacency_matrix = (int*)malloc(n * n * sizeof(int*));

		for (int i = 0; i < n; i++)
		{
			bool connected = false;

			for (int j = i; j < n; j++)
			{
				if (i == j) adjacency_matrix[i * n + j] = 0;
				else if (j == n - 1 && connected == false)
				{
					adjacency_matrix[i * n + j] = adjacency_matrix[j * n + i] = 1;
				}
				else
				{
					int r = rand() % 2;
					int val = (r == 0) ? 1 : 0;
					if (val == 1) connected = true;
					adjacency_matrix[i * n + j] = adjacency_matrix[j * n + i] = val;
				}
			}
		}

        // сохранение матрицы в файл
        if (save == 1)
        {
            std::cout << "Saving generated matrix to file \"adjacency_matrix.txt\"..." << std::endl << std::flush;
            std::ofstream path_file("adjacency_matrix.txt");
            path_file << "Adjacency matrix, size = " << n << "x" << n << "\n";
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    path_file << adjacency_matrix[i * n + j] << " ";
                }
                path_file << "\n";
            }
        }

		std::cout << std::endl << "Serial processing, please wait..." << std::endl << std::flush;
		start = MPI_Wtime();
		serial(n, adjacency_matrix, save);
		std::cout << "TIME: " << (MPI_Wtime() - start) << " seconds" << std::endl << std::endl;

		start = MPI_Wtime();
		std::cout << "Parallel processing, please wait..." << std::endl;
	}
	parallel(n, adjacency_matrix, rank, size, save);

	if (rank == 0)
	{
		std::cout << "TIME: " << (MPI_Wtime() - start) << " seconds" << std::endl;
	}

    free(adjacency_matrix);
	MPI_Finalize();
	return 0;
}
