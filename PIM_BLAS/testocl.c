#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include <sys/time.h>
#include <unistd.h>
#include <CL/cl.h>
#include "cblas.h"

#define CL_ERR_TO_STR(err) case err: return #err

double getthetime() {
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return (double)tv.tv_sec + (double)1.0e-6 * tv.tv_usec;
}


char const *
clGetErrorString(cl_int const err)
{
	switch(err)
	{
		CL_ERR_TO_STR(CL_SUCCESS);
		CL_ERR_TO_STR(CL_DEVICE_NOT_FOUND);
		CL_ERR_TO_STR(CL_DEVICE_NOT_AVAILABLE);
		CL_ERR_TO_STR(CL_COMPILER_NOT_AVAILABLE);
		CL_ERR_TO_STR(CL_MEM_OBJECT_ALLOCATION_FAILURE);
		CL_ERR_TO_STR(CL_OUT_OF_RESOURCES);
		CL_ERR_TO_STR(CL_OUT_OF_HOST_MEMORY);
		CL_ERR_TO_STR(CL_PROFILING_INFO_NOT_AVAILABLE);
		CL_ERR_TO_STR(CL_MEM_COPY_OVERLAP);
		CL_ERR_TO_STR(CL_IMAGE_FORMAT_MISMATCH);
		CL_ERR_TO_STR(CL_IMAGE_FORMAT_NOT_SUPPORTED);
		CL_ERR_TO_STR(CL_BUILD_PROGRAM_FAILURE);
		CL_ERR_TO_STR(CL_MAP_FAILURE);
		CL_ERR_TO_STR(CL_MISALIGNED_SUB_BUFFER_OFFSET);
		CL_ERR_TO_STR(CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST);
		CL_ERR_TO_STR(CL_COMPILE_PROGRAM_FAILURE);
		CL_ERR_TO_STR(CL_LINKER_NOT_AVAILABLE);
		CL_ERR_TO_STR(CL_LINK_PROGRAM_FAILURE);
		CL_ERR_TO_STR(CL_DEVICE_PARTITION_FAILED);
		CL_ERR_TO_STR(CL_KERNEL_ARG_INFO_NOT_AVAILABLE);
		CL_ERR_TO_STR(CL_INVALID_VALUE);
		CL_ERR_TO_STR(CL_INVALID_DEVICE_TYPE);
		CL_ERR_TO_STR(CL_INVALID_PLATFORM);
		CL_ERR_TO_STR(CL_INVALID_DEVICE);
		CL_ERR_TO_STR(CL_INVALID_CONTEXT);
		CL_ERR_TO_STR(CL_INVALID_QUEUE_PROPERTIES);
		CL_ERR_TO_STR(CL_INVALID_COMMAND_QUEUE);
		CL_ERR_TO_STR(CL_INVALID_HOST_PTR);
		CL_ERR_TO_STR(CL_INVALID_MEM_OBJECT);
		CL_ERR_TO_STR(CL_INVALID_IMAGE_FORMAT_DESCRIPTOR);
		CL_ERR_TO_STR(CL_INVALID_IMAGE_SIZE);
		CL_ERR_TO_STR(CL_INVALID_SAMPLER);
		CL_ERR_TO_STR(CL_INVALID_BINARY);
		CL_ERR_TO_STR(CL_INVALID_BUILD_OPTIONS);
		CL_ERR_TO_STR(CL_INVALID_PROGRAM);
		CL_ERR_TO_STR(CL_INVALID_PROGRAM_EXECUTABLE);
		CL_ERR_TO_STR(CL_INVALID_KERNEL_NAME);
		CL_ERR_TO_STR(CL_INVALID_KERNEL_DEFINITION);
		CL_ERR_TO_STR(CL_INVALID_KERNEL);
		CL_ERR_TO_STR(CL_INVALID_ARG_INDEX);
		CL_ERR_TO_STR(CL_INVALID_ARG_VALUE);
		CL_ERR_TO_STR(CL_INVALID_ARG_SIZE);
		CL_ERR_TO_STR(CL_INVALID_KERNEL_ARGS);
		CL_ERR_TO_STR(CL_INVALID_WORK_DIMENSION);
		CL_ERR_TO_STR(CL_INVALID_WORK_GROUP_SIZE);
		CL_ERR_TO_STR(CL_INVALID_WORK_ITEM_SIZE);
		CL_ERR_TO_STR(CL_INVALID_GLOBAL_OFFSET);
		CL_ERR_TO_STR(CL_INVALID_EVENT_WAIT_LIST);
		CL_ERR_TO_STR(CL_INVALID_EVENT);
		CL_ERR_TO_STR(CL_INVALID_OPERATION);
		CL_ERR_TO_STR(CL_INVALID_GL_OBJECT);
		CL_ERR_TO_STR(CL_INVALID_BUFFER_SIZE);
		CL_ERR_TO_STR(CL_INVALID_MIP_LEVEL);
		CL_ERR_TO_STR(CL_INVALID_GLOBAL_WORK_SIZE);
		CL_ERR_TO_STR(CL_INVALID_PROPERTY);
		CL_ERR_TO_STR(CL_INVALID_IMAGE_DESCRIPTOR);
		CL_ERR_TO_STR(CL_INVALID_COMPILER_OPTIONS);
		CL_ERR_TO_STR(CL_INVALID_LINKER_OPTIONS);
		CL_ERR_TO_STR(CL_INVALID_DEVICE_PARTITION_COUNT);

			default:
		return "UNKNOWN ERROR CODE";
	}
}

cl_int
cl_assert(cl_int const code, char const * const file, int const line, bool const abort)
{
	if (code != CL_SUCCESS)
	{
		char const * const err_str = clGetErrorString(code);

		fprintf(stderr, "\"%s\", line %d: cl_assert (%d) = \"%s\"", file, line, code, err_str);

		if (abort)
		{
			exit(code);
		}
	}

	return code;
}

#define MAX_SOURCE_SIZE (0x100000)

#define cl(...)		cl_assert((cl##__VA_ARGS__), __FILE__, __LINE__, true);
#define cl_ok(err)	cl_assert(err, __FILE__, __LINE__, true);

int main(void) {

  //C test code

   float x[36];
    for(int j=0; j<36; j++) {
        x[j] = 35-j;
    }

    float y[36];
    for(int i=0; i<36; i++) {
        y[i] = i;
    }

    float alpha = 2.0;
    float beta = 7.0;


//cblas_sscal
    cblas_sscal(18, alpha, y, 2);
    printf("OCL result of cblas_sscal\n");
    for(int i=0; i<36; i++) {
        //printf("%f \n", y[i]);
    }
    printf("\n");

//cblas_sdsdot
    cblas_sdsdot(18, alpha, x, 2, y, 2);
    printf("OCL result of cblas_sdsdot\n");
    for(int i=0; i<36; i++) {
        //printf("%f \n", y[i]);
    }
    printf("\n"); // sdsdot 삭제

//cblas_sdot
    cblas_sdot(18, x, 2, y, 2);
    printf("OCL result of cblas_sdot\n");
    for(int i=0; i<36; i++) {
        //printf("%f \n", y[i]);
    }
    printf("\n");

//cblas_saxpy
    cblas_saxpy(18, alpha, x, 2, y, 2);
    printf("OCL result of cblas_saxpy\n");
    for(int i=0; i<36; i++) {
        //printf("%f \n", y[i]);
    }
    printf("\n");

    float A1[36];
    for(int j=0; j<36; j++) {
        A1[j] = 7*(35-j);
    }
    float B1[36];
    for(int i=0; i<36; i++) {
        B1[i] = 3*i;
    }
    float C1[36];
    for(int i=0; i<36; i++) {
        C1[i] = 3*i;
    }
    float X[18];
    for(int i=0; i<18; i++) {
        X[i] = 9*i;
    }
    float Y[12];
    for(int i=0; i<12; i++) {
        Y[i] = 4*i;
    }

//cblas_sgemv
    cblas_sgemv(CblasColMajor, CblasNoTrans, 6, 6, alpha, A1, 6, X, 3, beta, Y, 2);
    printf("OCL result of cblas_sgemv\n");
    for(int i=0; i<12; i++) {
        //printf("%f \n", Y[i]);
    }
    printf("\n");
//cblas_sgemm
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 6, 6, 6, alpha, A1, 6, B1, 6, beta, C1, 6);
    printf("OCL result of cblas_sgemm\n");
    for(int i=0; i<36; i++) {
        //printf("%f \n", C1[i]);
    }
    printf("\n");
    printf("\n");
    

	// Create the two input vectors
	int N = 64;

	float *A = (float *)malloc(sizeof(float) * (N) * (N));
	float *B = (float *)malloc(sizeof(float) * (N) * (N));

	int i, j, k;

	for(i = 0; i < N; i++) {
		for(j = 0; j < N; j++) {
			A[i * N + j] = (rand() / (float)RAND_MAX) * (0.5 - 0) + 0.5;
			B[i * N + j] = (rand() / (float)RAND_MAX) * (0 - 0.5) - 0.5;
		}
	}

	// Load the kernel source code into the array source_str
	FILE *fp;
	char *source_str;
	size_t source_size;

	fp = fopen("cblas_sgemm.cl", "r");
	if (!fp) {
		fprintf(stderr, "Failed to load kernel.\n");
		exit(1);
	}
	source_str = (char*)malloc(MAX_SOURCE_SIZE);
	source_size = fread( source_str, 1, MAX_SOURCE_SIZE, fp);
	fclose( fp );

	// Get platform and device information
	cl_platform_id platform_id = NULL;
	cl_device_id device_id = NULL;   
	cl_uint ret_num_devices;
	cl_uint ret_num_platforms;
	
	cl_int ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
	cl_ok(ret);

	ret = clGetDeviceIDs( platform_id, CL_DEVICE_TYPE_GPU, 1, 
			&device_id, &ret_num_devices);
	cl_ok(ret);

	// Create an OpenCL context
	cl_context context = clCreateContext( NULL, 1, &device_id, NULL, NULL, &ret);
	cl_ok(ret);

	// Create a command queue
	cl_command_queue command_queue = clCreateCommandQueue(context, device_id, 0, &ret);
	cl_ok(ret);

	// Create memory buffers on the device for each vector 
	cl_mem a_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY, 
			N * N * sizeof(float), NULL, &ret);
	cl_ok(ret);

	cl_mem b_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY,
			N * N * sizeof(float), NULL, &ret);
	cl_ok(ret);

	cl_mem c_mem_obj = clCreateBuffer(context, CL_MEM_WRITE_ONLY, 
			N * N * sizeof(float), NULL, &ret);
	cl_ok(ret);

	// Copy the lists A and B to their respective memory buffers
	ret = clEnqueueWriteBuffer(command_queue, a_mem_obj, CL_TRUE, 0,
			N * N * sizeof(float), A, 0, NULL, NULL);
	cl_ok(ret);

	ret = clEnqueueWriteBuffer(command_queue, b_mem_obj, CL_TRUE, 0, 
			N * N * sizeof(float), B, 0, NULL, NULL);
	cl_ok(ret);

	clFinish(command_queue);

	// Create a program from the kernel source
	cl_program program = clCreateProgramWithSource(context, 1, 
			(const char **)&source_str, (const size_t *)&source_size, &ret);
	cl_ok(ret);

	// Build the program
	ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
	cl_ok(ret);

	// Create the OpenCL kernel
	cl_kernel kernel = clCreateKernel(program, "matmul_HW2", &ret);
	cl_ok(ret);

	// Set the arguments of the kernel
	ret = clSetKernelArg(kernel, 0, sizeof(int), &N);
	cl_ok(ret);
	ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&a_mem_obj);
	cl_ok(ret);
	ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&b_mem_obj);
	cl_ok(ret);
	ret = clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *)&c_mem_obj);
	cl_ok(ret);

	// Execute the OpenCL kernel on the list
	size_t global_item_size[2] = {N, N};

	double start_time, end_time;

	start_time = getthetime();
	ret = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, 
			global_item_size, NULL, 0, NULL, NULL);
	cl_ok(ret);

	end_time = getthetime();
	clFinish(command_queue);

	// Read the memory buffer C on the device to the local variable C
	float *C = (float *)malloc(sizeof(float) * N * N);
	ret = clEnqueueReadBuffer(command_queue, c_mem_obj, CL_TRUE, 0, 
			N * N * sizeof(float), C, 0, NULL, NULL);
	cl_ok(ret);

	// Result check, compare match
	float *C_ref = (float *)malloc(sizeof(float) * N * N);
	for(i = 0; i < N; i++) {
		for(j = 0; j < N; j++) {
			float Csub = 0.0f;
			for(k = 0; k < N; k++) {
				Csub += A[i * N + k] * B[j + N * k];
			}
			C_ref[i * N + j] = Csub;
		}
	}

	int res_check = 1;
	for(i = 0; i < N * N; i++) {
		if( (float)(fabs(C_ref[i] - C[i]) / fabs(C_ref[i]))>= 1e-6 ) {
			res_check = 0;
		}
	}

	printf("\nOpenCL Kernel Execution Time: %.3lf us\n\n", (end_time - start_time)*1000000);
  
  printf("testocl Successfully Executed !!! \n");


	// Clean up
	ret = clFlush(command_queue);
	ret = clFinish(command_queue);
	ret = clReleaseKernel(kernel);
	ret = clReleaseProgram(program);
	ret = clReleaseMemObject(a_mem_obj);
	ret = clReleaseMemObject(b_mem_obj);
	ret = clReleaseMemObject(c_mem_obj);
	ret = clReleaseCommandQueue(command_queue);
	ret = clReleaseContext(context);
	free(A);
	free(B);
	free(C);
	free(C_ref);
	return 0;
}
