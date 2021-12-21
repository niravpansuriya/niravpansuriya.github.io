%%cu
#include <iostream>
#include <ctime>
#include <cstdlib>
#define MIN_RANDOM_NUMBER 1
#define MAX_RANDOM_NUMBER 100
using namespace std;
#define BREAK 1

inline void set_matrix_element(float *matrix, int row, int column, int number_of_columns, float element)
{
	matrix[row *number_of_columns + column] = element;
}

inline float get_matrix_element(float *matrix, int row, int column, int number_of_columns)
{
	return matrix[row *number_of_columns + column];
}

__device__ float get_matrix_element_parallel(float *d_matrix, int row, int column, int number_of_columns)
{
	return d_matrix[row *number_of_columns + column];
}

inline void free_memory(float *ptr)
{
	free(ptr);
}

inline int generate_random_number(int min, int max)
{
	return rand() % (max - min + 1) + min;
}

inline void copy_array(float *src, float *dest, int size)
{
	for (int i = 0; i < size; i++)
		dest[i] = src[i];
}

void generate_weights(float *matrix, int rows, int columns)
{
	// initialize whole matrix with zero
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < columns; j++)
		{
			set_matrix_element(matrix, i, j, rows, 0);
		}
	}

	for (int i = 0; i < rows; i++)
	{
		long total = 0;
		for (int j = 0; j < columns; j++)
		{
			if (i == j)
				continue;
			int randomNumber = generate_random_number(MIN_RANDOM_NUMBER, MAX_RANDOM_NUMBER);
			set_matrix_element(matrix, i, j, rows, randomNumber);
			total += randomNumber;
		}

		int randomNumber = generate_random_number(MIN_RANDOM_NUMBER, MAX_RANDOM_NUMBER);
		set_matrix_element(matrix, i, i, rows, total + randomNumber);
	}
}

void generate_constants(float *vector, int size)
{
	// initialize vector with zero
	for (int i = 0; i < size; i++)
	{
		vector[i] = 0;
	}

	for (int i = 0; i < size; i++)
	{
		int randomNumber = generate_random_number(MIN_RANDOM_NUMBER, MAX_RANDOM_NUMBER);
		vector[i] = randomNumber;
	}
}

void initialize_X(float *X, int size)
{
	for (int i = 0; i < size; i++)
	{
		X[i] = 0;
	}
}

__global__ void initialize_x_parallel(float *d_X, int size)
{
	// get the thread id
	int i = blockIdx.x *blockDim.x + threadIdx.x;

	if (i < size) d_X[i] = 0;
}

bool is_convergence(float *weights, float *X, float *constants, int size, float threshold)
{
	float error = 0;

	for (int i = 0; i < size; i++)
	{
		float prediction = 0;
		for (int j = 0; j < size; j++)
		{
			prediction += get_matrix_element(weights, i, j, size) *X[j];
		}

		error += abs(prediction - constants[i]);
	}

	if (error <= threshold)
		return true;
	return false;
}

__global__ void predict_output_parallel(float *d_weights, float *d_X, float *constants, float *d_prediction, float size)
{
	int i = blockIdx.x *blockDim.x + threadIdx.x;

	if (i < size)
	{
		float prediction = 0;

		for (int j = 0; j < size; j++)
		{
			prediction += get_matrix_element_parallel(d_weights, i, j, size) *d_X[j];
		}

		d_prediction[i] = prediction;
	}
}

// Will do array1 = abs(array1 - array2)
__global__ void subtract_array_parallel(float *array1, float *array2, int size)
{
	int i = blockIdx.x *blockDim.x + threadIdx.x;

	if (i < size)
	{
		array1[i] -= array2[i];
		if (array1[i] < 0) array1[i] = abs(array1[i]);
	}
}

__global__ void execute_sum(float *array, float *answer, int size)
{
	int i = blockIdx.x *blockDim.x + threadIdx.x;

	if (i < size)
	{
		int iterations = ceil(__log2f(size));
		int count = 0;
		int n = 1;

		while (count != iterations)
		{
			n *= 2;

			if (i % n == 0 && i + (n / 2) < size)
			{
				array[i] += array[i + (n / 2)];
			}

			count++;
			__syncthreads();
		}

		if (i == 0)
		{
			*answer = array[0];
		}
	}
}

void array_sum_parallel(float *d_array, float *answer, int size)
{
	float *d_answer;
	cudaMalloc(&d_answer, sizeof(float));
	execute_sum <<<size / 256 + 1, 256>>> (d_array, d_answer, size);
	cudaMemcpy(answer, d_answer, sizeof(float), cudaMemcpyDeviceToHost);

	cudaFree(d_answer);
}


bool check_convergence_parallel(float *d_weights, float *d_X, float *d_constants, float *d_prediction, int size, float threshold)
{
	int blocks = size / 256 + 1;
	predict_output_parallel <<<blocks, 256>>> (d_weights, d_X, d_constants, d_prediction, size);

	subtract_array_parallel <<<blocks, 256>>> (d_prediction, d_constants, size);

	float *error = (float*) malloc(sizeof(float));

	array_sum_parallel(d_prediction, error, size);

	if (*error <= threshold) return true;
	return false;
}

void print_matrix(float *matrix, int number_of_rows, int number_of_columns)
{
	for (int i = 0; i < number_of_rows; i++)
	{
		for (int j = 0; j < number_of_columns; j++)
		{
			cout << get_matrix_element(matrix, i, j, number_of_columns) << " ";
		}

		cout << endl;
	}
}

void jacobian(float *weights, float *X, float *constants, int size)
{
	cout << "Jacobian method---------------------------------------------------" << endl;
	float *X_prev = (float*) malloc(size* sizeof(float));
	initialize_X(X, size);
	copy_array(X, X_prev, size);

	long iterations = 0;
	while (!is_convergence(weights, X, constants, size, 0.005))
	{
		for (int i = 0; i < size; i++)
		{
			X[i] = constants[i];
			for (int j = 0; j < size; j++)
			{
				if (i == j)
					continue;
				X[i] -= get_matrix_element(weights, i, j, size) *X_prev[j];
			}

			X[i] /= get_matrix_element(weights, i, i, size);
		}

		copy_array(X, X_prev, size);
		iterations++;
	}

	cout << endl;

	cout << "variables..." << endl;
	print_matrix(X, size, 1);
	cout << endl;

	cout << "iterations..." << iterations << endl;
	free_memory(X_prev);

	cout << "--------------------------------------------------------------------" << endl;
}

void PJG(float *weights, float *X, float *constants, int P, int size)
{
	cout << "PJG method---------------------------------------------------" << endl;
	float *X_prev = (float*) malloc(P* sizeof(float));
	initialize_X(X, size);
	initialize_X(X_prev, P);

	long iterations = 0;
	while (!is_convergence(weights, X, constants, size, 0.005))
	{
		int blockIndex = 0;
		int start = 0;
		int end = 0;
		while(end != size){
			start = blockIndex*P;
			end = start + P;
			
			if(end>size)	end = size;
			
			int count = 0;
			for (int i = start; i < end; i++)
			{
				X_prev[count] = constants[i];
				for (int j = 0; j < size; j++)
				{
					if (i == j)
						continue;
					X_prev[count] -= get_matrix_element(weights, i, j, size) *X[j];
				}

				X_prev[count] /= get_matrix_element(weights, i, i, size);
				count++;
			}
			
			count = 0;
			// update X
			for(int i=start;i<end;i++){
				X[i] = X_prev[count++];
			}
			blockIndex++;
		}

		iterations++;

		if(iterations == BREAK)	break;

	}

	cout << endl;

	cout << "variables..." << endl;
	print_matrix(X, size, 1);
	cout << endl;

	cout << "iterations..." << iterations << endl;
	free_memory(X_prev);

	cout << "--------------------------------------------------------------------" << endl;
}

__global__ void PJG_kernel(float *d_weights, float *d_X, float *d_constants, int P, int size){

	int threadId = blockIdx.x * blockDim.x + threadIdx.x;

	if(threadId < P){
		int blockIndex = 0;
		int i = threadId;
		float prediction = 0;
		while(i<size){
			prediction = d_constants[i];

			for(int j=0;j<size;j++){
				if(i==j)	continue;
				prediction -= get_matrix_element_parallel(d_weights, i, j, size) * d_X[j];
			}

			prediction /= get_matrix_element_parallel(d_weights, i, i, size);

			d_X[i] = prediction;
			blockIndex++;
			i = blockIndex * P + threadId;
			__syncthreads();
		}
	}
}



void PJG_parallel(float *weights, float *X, float *constants, int P, int size)
{
	cout << "PJG parallel method---------------------------------------------------" << endl;
	
	int blocks = P/16+1;
	int threads = 16;

	// transfer weights to device
	float *d_weights;
	cudaMalloc(&d_weights, sizeof(float) * size * size);
	cudaMemcpy(d_weights, weights, sizeof(float) * size * size, cudaMemcpyHostToDevice);

	// transfer X to device
	float *d_X;
	cudaMalloc(&d_X, sizeof(float)*size);
	initialize_x_parallel<<<blocks, threads>>>(d_X, size);

	// transfer constants to device
	float *d_constants;
	cudaMalloc(&d_constants, sizeof(float)*size);
	cudaMemcpy(d_constants, constants, sizeof(float)*size, cudaMemcpyHostToDevice);

	// define array for check convergence
	float *d_predictions;
	cudaMalloc(&d_predictions, sizeof(float)*size);

	bool is_convergence = false;
	long iterations = 0;
	while (!is_convergence)
	{
		
		PJG_kernel<<<blocks, threads>>>(d_weights, d_X, d_constants, P, size);
	
		iterations++;
		
		is_convergence = check_convergence_parallel(d_weights, d_X, d_constants, d_predictions, size, 0.005);

	}

	// transfer variables X to host
	cudaMemcpy(X, d_X, sizeof(float)*size, cudaMemcpyDeviceToHost);
	cout << endl;

	cout << "variables..." << endl;
	print_matrix(X, size, 1);
	cout << endl;

	cout << "iterations..." << iterations << endl;

	// destroy memory in device
	cudaFree(d_weights);
	cudaFree(d_X);
	cudaFree(d_constants);
	cudaFree(d_predictions);

	cout << "--------------------------------------------------------------------" << endl;
}




void gauss_seidel(float *weights, float *X, float *constants, int size)
{
	cout << "Gauss Seidel method---------------------------------------------------" << endl;
	initialize_X(X, size);

	long iterations = 0;
	while (!is_convergence(weights, X, constants, size, 0.005))
	{
		for (int i = 0; i < size; i++)
		{
			X[i] = constants[i];
			for (int j = 0; j < size; j++)
			{
				if (i == j)
					continue;
				X[i] -= get_matrix_element(weights, i, j, size) *X[j];
			}

			X[i] /= get_matrix_element(weights, i, i, size);
		}

		iterations++;
	}

	cout << endl;

	cout << "variables..." << endl;
	print_matrix(X, size, 1);
	cout << endl;

	cout << "iterations..." << iterations << endl;

	cout << "--------------------------------------------------------------------" << endl;
}

__global__ void outer_calculation_parallel(float *d_weights, float *d_X, float *d_X_prev, float *d_constants, int size)
{
	int i = blockIdx.x *blockDim.x + threadIdx.x;

	if (i < size)
	{
		d_X[i] = d_constants[i];
		for (int j = 0; j < size; j++)
		{
			if (i == j) continue;

			d_X[i] -= get_matrix_element_parallel(d_weights, i, j, size) *d_X_prev[j];
		}

		d_X[i] /= get_matrix_element_parallel(d_weights, i, i, size);
	}
}

__global__ void inner_calculation_parallel(float *d_weights, float *d_X, float *d_constants, float *d_multiplications, int i, int size)
{
	int j = blockIdx.x *blockDim.x + threadIdx.x;

	if(i == j)	d_multiplications[j]=0;
	else if (j < size)
	{
		d_multiplications[j] = get_matrix_element_parallel(d_weights, i, j, size) *d_X[j];
	}
}

__global__ void calculate_x_for_row_based_parallel(float *d_weights, float *d_X, float *d_constants, float *d_sum, int i, int size)
{
	int j = blockIdx.x *blockDim.x + threadIdx.x;

	if(j==0){
		d_X[i] = (d_constants[i]-*d_sum)/get_matrix_element_parallel(d_weights, i, i, size);
	}
}

__global__ void PJG_kernel_improved(float *d_weights, float *d_X, float *d_constants, float *d_multiplications, int P, int size){

	int threadNumber = blockIdx.x * blockDim.x + threadIdx.x;
	int threadId = threadNumber/size;

	if(threadId < P){
		int blockIndex = 0;
		int i = threadId;
		float d_sum = 0;
		int j = threadNumber%size;

		while(i<size){
			if(i==j)	d_multiplications[threadNumber] = 0;
			else d_multiplications[threadNumber] = get_matrix_element_parallel(d_weights,i,j,size) * d_X[j];
			__syncthreads();

			// summation
			int iterations = ceil(__log2f(size));
			int n=1;
			int count = 0;

			while(count!=iterations){
				n*=2;
				if(j%n==0 && j+(n/2)<size){
					d_multiplications[threadNumber] += d_multiplications[threadNumber+(n/2)];
				}
				count++;
				__syncthreads();
			}


			if(j==0){
				// float sum = 0;
				// for(int x=0;x<size;x++){
				// 	sum += d_multiplications[threadNumber + x];
				// }
				d_X[i] = (d_constants[i]-d_multiplications[threadNumber])/get_matrix_element_parallel(d_weights,i,i,size);
				// d_X[i] = (d_constants[i]-sum)/get_matrix_element_parallel(d_weights,i,i,size);
			}

			blockIndex++;
			i = blockIndex * P + threadId;
			__syncthreads();
		}
	}
}


void jacobianParallelMethod(float *weights, float *X, float *constants, int size, bool shouldExchangeError)
{
	int blocks = (size / 256) + 1;
	int threads = 256;
	if (shouldExchangeError)
	{
		cout << "Jacobian Parallel Method with Exchange Error..." << endl;

		float *d_X_prev, *d_X, *d_weights, *d_constants, *d_predictions;
		cudaMalloc(&d_weights, sizeof(float) *size *size);
		cudaMemcpy(d_weights, weights, sizeof(float) *size *size, cudaMemcpyHostToDevice);
        
        cudaMalloc(&d_constants, sizeof(float)*size);
        cudaMemcpy(d_constants, constants, sizeof(float)*size, cudaMemcpyHostToDevice);

		cudaMalloc(&d_X_prev, sizeof(float) *size);
		cudaMalloc(&d_X, sizeof(float) *size);
		cudaMalloc(&d_predictions, sizeof(float) *size);

		initialize_x_parallel <<<size / 256 + 1, 256>>> (d_X_prev, size);
		initialize_x_parallel <<<size / 256 + 1, 256>>> (d_X, size);
        
		long iterations = 0;
		bool is_convergence = false;
	
		while (!is_convergence)
		{
			if (iterations % 2 == 0)
			{
				outer_calculation_parallel<<<blocks, threads>>>(d_weights, d_X, d_X_prev, d_constants, size);
                is_convergence = check_convergence_parallel (d_weights, d_X, d_constants, d_predictions, size, 0.005);

			}
			else
			{
				outer_calculation_parallel<<<blocks, threads>>>(d_weights, d_X_prev, d_X, d_constants, size);
				is_convergence = check_convergence_parallel (d_weights, d_X_prev, d_constants, d_predictions, size, 0.005);

			}

			iterations++;
		}

        cout<<endl;
        cout<<"Execution complete..."<<endl;

        if(iterations % 2 == 0){
            cudaMemcpy(X, d_X_prev, sizeof(float)*size, cudaMemcpyDeviceToHost);
        }
        else{
            cudaMemcpy(X, d_X, sizeof(float)*size, cudaMemcpyDeviceToHost);
        }

        print_matrix(X, size, 1);
        cout<<iterations<<endl;
		cudaFree(d_X_prev);
		cudaFree(d_X);
		cudaFree(d_weights);
		cudaFree(d_constants);
	}
}

void PJG_parallel_improved(float *weights, float *X, float *constants, int P, int size)
{
	cout << "PJG parallel improved method---------------------------------------------------" << endl;
	
	int blocks = (P*size)/16+1;
	int threads = 16;

	// transfer weights to device
	float *d_weights;
	cudaMalloc(&d_weights, sizeof(float) * size * size);
	cudaMemcpy(d_weights, weights, sizeof(float) * size * size, cudaMemcpyHostToDevice);

	// transfer X to device
	float *d_X;
	cudaMalloc(&d_X, sizeof(float)*size);
	initialize_x_parallel<<<blocks, threads>>>(d_X, size);

	// transfer constants to device
	float *d_constants;
	cudaMalloc(&d_constants, sizeof(float)*size);
	cudaMemcpy(d_constants, constants, sizeof(float)*size, cudaMemcpyHostToDevice);

	// define array for check convergence
	float *d_predictions;
	cudaMalloc(&d_predictions, sizeof(float)*size);

	// define multiplication matrix in device
	float *d_multiplications;
	cudaMalloc(&d_multiplications, sizeof(float)*size*P);
	
	bool is_convergence = false;
	long iterations = 0;
	while (!is_convergence)
	{
		
		PJG_kernel_improved<<<blocks, threads>>>(d_weights, d_X, d_constants, d_multiplications, P, size);
	
		iterations++;
		
		is_convergence = check_convergence_parallel(d_weights, d_X, d_constants, d_predictions, size, 0.005);

		if(iterations == BREAK)	break;
	}

	// transfer variables X to host
	cudaMemcpy(X, d_X, sizeof(float)*size, cudaMemcpyDeviceToHost);
	cout << endl;

	cout << "variables..." << endl;
	print_matrix(X, size, 1);
	cout << endl;

	cout << "iterations..." << iterations << endl;

	// destroy memory in device
	cudaFree(d_weights);
	cudaFree(d_X);
	cudaFree(d_constants);
	cudaFree(d_predictions);

	cout << "--------------------------------------------------------------------" << endl;
}

void rowBasedParallelMethod(float *weights, float *X, float *constants, int size){
	int blocks = (size / 256) + 1;
	int threads = 256;

	cout << "Row Based Parallel Method..." << endl;

	float  *d_X, *d_weights, *d_constants, *d_predictions, *d_multiplications, *d_sum;

	cudaMalloc(&d_weights, sizeof(float) *size *size);
	cudaMemcpy(d_weights, weights, sizeof(float) *size *size, cudaMemcpyHostToDevice);
      
	cudaMalloc(&d_constants, sizeof(float)*size);
	cudaMemcpy(d_constants, constants, sizeof(float)*size, cudaMemcpyHostToDevice);

	cudaMalloc(&d_multiplications, sizeof(float)*size);
	cudaMalloc(&d_sum, sizeof(float));
	
	cudaMalloc(&d_X, sizeof(float) *size);
	cudaMalloc(&d_predictions, sizeof(float) *size);
	
	initialize_x_parallel <<<size / 256 + 1, 256>>> (d_X, size);

	bool is_convergence = false;
	long iterations = 0;
	while(!is_convergence){
		for(int i=0;i<size;i++){
			inner_calculation_parallel<<<blocks, threads>>>(d_weights, d_X, d_constants, d_multiplications, i, size);
			execute_sum <<<blocks, threads>>> (d_multiplications, d_sum, size);
			calculate_x_for_row_based_parallel<<<1,1>>>(d_weights, d_X, d_constants, d_sum, i, size);
		}
		is_convergence = check_convergence_parallel(d_weights, d_X, d_constants, d_predictions, size, 0.005);
		iterations++;
	}
    cudaMemcpy(X, d_X, sizeof(float)*size, cudaMemcpyDeviceToHost);
	print_matrix(X, size, 1);
	cout<<"iterations "<<iterations<<endl;
	cudaFree(d_X);
	cudaFree(d_weights);
	cudaFree(d_constants);
	cudaFree(d_multiplications);
	cudaFree(d_sum);
}

int main()
{
	int size = 10;
	int P = 3;
	float *weights = (float*) malloc(size *size* sizeof(float));
	float *constants = (float*) malloc(size* sizeof(float));
	float *X = (float*) malloc(size* sizeof(float));

	cout << "weights..." << endl;
	generate_weights(weights, size, size);
	// print_matrix(weights, size, size);
	cout << endl;

	cout << "constants..." << endl;
	generate_constants(constants, size);
	// print_matrix(constants, size, 1);
	cout << endl;



	// jacobian(weights, X, constants, size);
	PJG(weights, X, constants, P, size);
	PJG_parallel_improved(weights, X, constants, P, size);
    // jacobianParallelMethod(weights, X, constants, size, true);
	// gauss_seidel(weights, X, constants, size);
	// rowBasedParallelMethod(weights, X, constants, size);
	free_memory(weights);
	free_memory(constants);
	free_memory(X);
	return 0;
}