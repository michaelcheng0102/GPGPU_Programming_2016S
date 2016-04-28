#include "lab3.h"
#include <cstdio>

__device__ __host__ int CeilDiv(int a, int b) { return (a-1)/b + 1; }
__device__ __host__ int CeilAlign(int a, int b) { return CeilDiv(a, b) * b; }

__global__ void SimpleClone(
	const float *background,
	const float *target,
	const float *mask,
	float *output,
	const int wb, const int hb, const int wt, const int ht,
	const int oy, const int ox
) {
	const int yt = blockIdx.y * blockDim.y + threadIdx.y;
	const int xt = blockIdx.x * blockDim.x + threadIdx.x;
	const int curr = wt*yt+xt;
	if (0 <= yt && 0 <= xt && yt < ht && xt < wt && mask[curr] > 127.0) {
		const int yb = oy+yt, xb = ox+xt;
		const int curb = wb*yb+xb;
		if (0 <= yb && yb < hb && 0 <= xb && xb < wb) {
			output[curb*3+0] = target[curr*3+0];
			output[curb*3+1] = target[curr*3+1];
			output[curb*3+2] = target[curr*3+2];
		}
	}
}


__global__ void CalculateFixed(
	const float *background,
	const float *target,
	const float *mask,
	float *fixed,
	const int wb, const int hb, const int wt, const int ht,
	const int oy, const int ox
) {
	const int yt = blockIdx.y * blockDim.y + threadIdx.y;
	const int xt = blockIdx.x * blockDim.x + threadIdx.x;
	const int curr = wt*yt+xt;
	const int yb = oy+yt;
	const int xb = ox+xt;
	const int curb = wb*yb+xb;
	int targetN, targetW, targetS, targetE;
	int boardN, boardW, boardS, boardE;
	/*
	 * Check its in target
	 */
	if(0 <= yb && 0 <= xb && yb < hb && xb < wb) {
		/*
		 * If the mask is black, should set to the background
		 * Else, the fixed part should be calculated with the target and background together
		 */
		if(mask[curr] < 127.0) {
			fixed[3*curr] = background[3*curb];
			fixed[3*curr+1] = background[3*curb+1];
			fixed[3*curr+2] = background[3*curb+2];
		}
		else {
			/*
			 * Calculate the N, S, E, W pixel of the target and board
			 */
			targetN = wt*(yt-1)+xt;
			boardN = wb*(yb-1)+xb;
			targetW = wt*yt+(xt-1);
			boardW = wb*yb+(xb-1);
			targetS = wt*(yt+1)+xt;
			boardS = wb*(yb+1)+xb;
			targetE = wt*yt+(xt+1);
			boardE = wb*yb+(xb+1);

			for(int rgb = 0; rgb < 3; rgb++) {
				/*
				 * Calaulate target fixed compaer with N, W, S, E
				 */
				float targetCalculation = 0.0;
				float borderCalculation = 0.0;
				if(yt > 0) {
					targetCalculation += target[3*curr+rgb] - target[3*targetN+rgb];
				}
				if(xt > 0) {
					targetCalculation += target[3*curr+rgb] - target[3*targetW+rgb];
				}
				if(yt < ht-1) {
					targetCalculation += target[3*curr+rgb] - target[3*targetS+rgb];
				}
				if(xt < wt-1)  {
					targetCalculation += target[3*curr+rgb] - target[3*targetE+rgb];
				}

				/*
				 * Calculate border and mask is black in N, W, E, S
				 */
				if(yt == 0 || mask[targetN] < 127.0) {
					borderCalculation += background[3*boardN+rgb];
				}
				if(xt == 0 || mask[targetW] < 127.0) {
					borderCalculation += background[3*boardW+rgb];
				}
				if(yt == ht-1 || mask[targetS] < 127.0) {
					borderCalculation += background[3*boardS+rgb];
				}
				if(xt == wt-1 || mask[targetE] < 127.0) {
					borderCalculation += background[3*boardE+rgb];
				}
				/*
				 * The fixed part is the combination of border and target calculation
				 */
				fixed[3*curr+rgb] = targetCalculation + borderCalculation;
			}
		}
	}
}


__global__ void PoissonImageCloningIteration(
	float *fixed,
	const float *mask,
	float *buf1, float *buf2,
	const int wt, const int ht
) {
	const int yt = blockIdx.y * blockDim.y + threadIdx.y;
	const int xt = blockIdx.x * blockDim.x + threadIdx.x;
	const int curr = wt*yt+xt;
	int targetN, targetS, targetE, targetW;
	float sum;
	/*
	 * Check if the target pixel is in the border
	 */
	if (0 <= yt && 0 <= xt && yt < ht && xt < wt && mask[curr] > 127.0) {
		targetN = wt*(yt-1)+xt;
		targetW = wt*yt+(xt-1);
		targetS = wt*(yt+1)+xt;
		targetE = wt*yt+(xt+1);
		for(int rgb = 0; rgb < 3; rgb++) {
			sum = 0.0;
			if(yt > 0 && mask[targetN] > 127.0) {
				sum += buf1[3*targetN+rgb];
			}
			if(xt > 0 && mask[targetW] > 127.0) {
				sum += buf1[3*targetW+rgb];
			}
			if(yt < ht-1 && mask[targetS] > 127.0) {
				sum += buf1[3*targetS+rgb];
			}
			if(xt < wt-1 && mask[targetE] > 127.0) {
				sum += buf1[3*targetE+rgb];
			}
			// printf("current mask: %d", sum);
			buf2[3*curr+rgb] = (sum + fixed[curr*3+rgb]) / 4.0;
		}
	}
}


void PoissonImageCloning(
	const float *background,
	const float *target,
	const float *mask,
	float *output,
	const int wb, const int hb, const int wt, const int ht,
	const int oy, const int ox
) {
	float *fixed, *buf1, *buf2;
	cudaMalloc(&fixed, 3*wt*ht*sizeof(float));
	cudaMalloc(&buf1, 3*wt*ht*sizeof(float));
	cudaMalloc(&buf2, 3*wt*ht*sizeof(float));

	dim3 gdim(CeilDiv(wt, 32), CeilDiv(ht, 16)), bdim(32, 16);

	CalculateFixed<<<gdim, bdim>>>(
		background, target, mask, fixed,
		wb, hb, wt, ht, oy, ox
	);
	cudaMemcpy(buf1, target, sizeof(float)*3*wt*ht, cudaMemcpyDeviceToDevice);

	for(int i = 0; i < 10000; i++) {
		PoissonImageCloningIteration<<<gdim, bdim>>>(
			fixed, mask, buf1, buf2, wt, ht
		);
		PoissonImageCloningIteration<<<gdim, bdim>>>(
			fixed, mask, buf2, buf1, wt, ht
		);
	}

	cudaMemcpy(output, background, wb*hb*sizeof(float)*3, cudaMemcpyDeviceToDevice);

	SimpleClone<<<gdim, bdim>>>(
		background, buf1, mask, output,
		wb, hb, wt, ht, oy, ox
	);

	cudaFree(fixed);
	cudaFree(buf1);
	cudaFree(buf2);

}
