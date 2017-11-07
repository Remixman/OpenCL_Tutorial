#include <stdio.h>
#include <stdlib.h>
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include <CL/cl.h>

#define MAX_KERNEL_SIZE 100000

int main() {
  int i;
  double *a, *b, *c, sum;
  size_t n = 1000000;
  size_t global_dim, local_dim;

  cl_mem d_a, d_b, d_c;
  cl_platform_id platform_id = NULL;
  cl_device_id device_id = NULL;
  cl_context context = NULL;
  cl_command_queue queue = NULL;
  cl_program program = NULL;
  cl_kernel vecadd_kernel = NULL;
  cl_int cl_ret;

  FILE *vecadd_kernel_file;
  char *kernel_source;
  size_t kernel_size;

  a = (double*) malloc(n * sizeof(double));
  b = (double*) malloc(n * sizeof(double));
  c = (double*) malloc(n * sizeof(double));

  for (i = 0; i < n; ++i) { a[i] = 3.5, b[i] = 4.5; }

  // Open kernel file and get source
  vecadd_kernel_file = fopen("vecadd.cl", "r");
  if (!vecadd_kernel_file) {
    fprintf(stderr, "Not found kernel file : vecadd.cl\n");
    return -1; 
  }
  kernel_source = (char*) malloc(MAX_KERNEL_SIZE * sizeof(char));
  fread(kernel_source, sizeof(char), MAX_KERNEL_SIZE * sizeof(char), vecadd_kernel_file);
  fclose(vecadd_kernel_file);

  // Number of work item (global_dim) and work group (local_dim)
  local_dim = 4;
  global_dim = ((n + local_dim - 1) / local_dim) * local_dim;

  cl_ret = clGetPlatformIDs(1, &platform_id, NULL);
  cl_ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, NULL);
  
  context = clCreateContext(0, 1, &device_id, 0, 0, &cl_ret);
  queue = clCreateCommandQueue(context, device_id, 0, &cl_ret);
  
  // Allocate device memory
  d_a = clCreateBuffer(context, CL_MEM_READ_ONLY, n * sizeof(double), NULL, &cl_ret);
  d_b = clCreateBuffer(context, CL_MEM_READ_ONLY, n * sizeof(double), NULL, &cl_ret);
  d_c = clCreateBuffer(context, CL_MEM_WRITE_ONLY, n * sizeof(double), NULL, &cl_ret);
  
  // Create program and kernel
  program = clCreateProgramWithSource(context, 1, (const char **)&kernel_source, 
                NULL, &cl_ret);
  cl_ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
  vecadd_kernel = clCreateKernel(program, "vecadd", &cl_ret);

  // Set kernel's arguments
  cl_ret = clSetKernelArg(vecadd_kernel, 0, sizeof(cl_mem), (void*)&d_a);
  cl_ret = clSetKernelArg(vecadd_kernel, 1, sizeof(cl_mem), (void*)&d_b);
  cl_ret = clSetKernelArg(vecadd_kernel, 2, sizeof(cl_mem), (void*)&d_c);
  cl_ret = clSetKernelArg(vecadd_kernel, 3, sizeof(cl_int), (void*)&n);

  // Transfer data from host to device
  cl_ret = clEnqueueWriteBuffer(queue, d_a, CL_TRUE, 0, sizeof(double) * n,
                a, 0, NULL, NULL);
  cl_ret = clEnqueueWriteBuffer(queue, d_b, CL_TRUE, 0, sizeof(double) * n,
                b, 0, NULL, NULL);

  // Execute the kernel
  cl_ret = clEnqueueNDRangeKernel(queue, vecadd_kernel, 1, NULL, 
                &global_dim, &local_dim, 0, NULL, NULL);

  // Wait command queue to finish
  cl_ret = clFinish(queue);

  // Transfer back thr result from device
  cl_ret = clEnqueueReadBuffer(queue, d_c, CL_TRUE, 0, sizeof(double) * n,
                c, 0, NULL, NULL);

  // Release device resources
	cl_ret = clReleaseMemObject(d_a);
	cl_ret = clReleaseMemObject(d_b);
  cl_ret = clReleaseMemObject(d_c);
  cl_ret = clReleaseCommandQueue(queue);
	cl_ret = clReleaseKernel(vecadd_kernel);
	cl_ret = clReleaseProgram(program);
	cl_ret = clReleaseContext(context);

  // Print result summation
  sum = 0.0;
  for (i = 0; i < n; ++i) sum += c[i];
  printf("Result summation : %.2lf\n", sum);

  // Release host resources
  free(kernel_source);
  free(a); free(b); free(c);

  return 0;
}
