#include "lab2.h"
#include <math.h>

#define PI 3.141592653589


static const unsigned W = 640;
static const unsigned H = 480;
static const unsigned NFRAME = 360;
__device__ const unsigned w = 640;
__device__ const unsigned h = 480;
__device__ const unsigned n = 360;

struct Lab2VideoGenerator::Impl {
	int t = 0;
};

Lab2VideoGenerator::Lab2VideoGenerator(): impl(new Impl) {
}

Lab2VideoGenerator::~Lab2VideoGenerator() {}

void Lab2VideoGenerator::get_info(Lab2VideoInfo &info) {
	info.w = W;
	info.h = H;
	info.n_frame = NFRAME;
	// fps = 24/1 = 24
	info.fps_n = 24;
	info.fps_d = 1;
};

__global__ void setPixelY(int time, uint8_t *yuv) {
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	int x = idx % w;
	int y = idx / w;
	int weight = (x+y) / (w+h);
	//*(yuv+idx) = weight*time*255/n;
	yuv[idx] = ( (idx%w) * 255 * abs(cos(time*PI/180.0)) ) / ( w );
}

void Lab2VideoGenerator::Generate(uint8_t *yuv) {
	//cudaMemset(yuv, (impl->t)*255/NFRAME, W*H);
	setPixelY<<<640, 480>>>(impl->t, yuv);
	cudaMemset(yuv+W*H, 128, W*H/2);
	++(impl->t);
}
