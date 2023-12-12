// HW_2
__kernel void matmul_HW2( 
		const int N,
		const __global float *A, 
		const __global float *B, 
		__global float *C)
{
	int tidx = get_global_id(0); // i
	int tidy = get_global_id(1); // j

	if (tidx < N && tidy < N)
	{
		float Csub = 0.0f;
		for(int k = 0; k < N; k++) // k
			Csub += A[ tidx * N + k ] * B[ k * N + tidy ];

		C[ tidx * N + tidy ] = Csub;
	}
}


