#include "lab2.h"
#include <time.h>
#include <math.h>

#define PI 3.141592653589
#define S 1000
#define fps 24

static const unsigned W = 640;
static const unsigned H = 480;
static const unsigned NFRAME = 480;
__device__ const unsigned w = 640;
__device__ const unsigned h = 480;
__device__ const unsigned n = 480;
static int flag = 0;
int *tempx;
int *tempy;
__device__ int error[S];
__device__ int err_count = 0;


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
	//*(yuv+idx) = weight*time*255/n;
	//printf("idx: %d, x:%d, y:%d\n", idx, x, y);
	yuv[idx] = ( (idx%w) * 255 * abs(cos(time*PI/180.0))) / ( w );
}

void generate_star() {
	srand(time(NULL));
	tempx = (int *)malloc(S*sizeof(int));
	tempy = (int *)malloc(S*sizeof(int));
	for(int i=0; i<S; i++) {
		tempx[i] = rand() % (W/4);
		tempy[i] = rand() % (H/4);
		printf("%d   x=%d, y=%d\n", i, tempx[i], tempy[i]);
	}
}

__global__ void test(int *x, int *y) {
	for(int i=0; i<S; i++) {
		printf("%d   x=%d y=%d\n", i, x[i], y[i]);
	}
}

__device__ int hash_size(int x, int y) {
	if((x+y)%16 == 0) return 4;
	if((x+y)%16 <= 2) return 3;
	if((x+y)%16 <= 6) return 2;
	else return 1;
}

__global__ void paintStar(int *x, int *y, int time, uint8_t *yuv) {
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if(idx > S) return;
	if(x[idx]*4+y[idx]*4*w >= w*h) {
		error[err_count]=idx;
		err_count ++;
	}
	for(int i=0; i<err_count; i++) {
		if(idx == error[i]) return;
	}

	int len = hash_size(x[idx], y[idx]);
	for(int i=0; i<len; i++) {
		for(int j=0; j<len; j++) {
			yuv[ x[idx]*4 + y[idx]*4*w + i + j*w ] = 255;
		}
	}
	//yuv[ x[idx]*4 + y[idx]*4*w  ] = 128 + 50*cos( cos(time*1.0)/10.0 + cos(1.0*x[idx]/w) + cos(1.0*y[idx]/h) );
	//yuv[ x[idx]*2 + y[idx]*2*w +1 ] = 255;
	//yuv[ x[idx]*2 + y[idx]*2*w +w ] = 255;
	//yuv[ x[idx]*2 + y[idx]*2*w +w+1 ] = 255;
}

__device__ void draw(int x1, int y1, uint8_t *yuv) {
	if(x1>=0 && x1<w && y1>=0 && y1<h)
		yuv[x1+y1*w] = 255;
}

__global__ void printStripe(int *x, int *y, int time, uint8_t *yuv, int len) {
	int idx = blockIdx.x+blockDim.x + threadIdx.x;
	if(idx > S) return;
	int x1 = x[idx]*4;
	int y1 = y[idx]*4;
	int dx = x1 - w/2;
	int dy = y1 - h/2;
	if(abs(dx) > abs(dy)) {
		if(x1 < w/2) {
			for(int i=x1; i>(x1 - len); i--)
				draw(i, y1 + dy * (i - x1) / dx, yuv);
		} else {
			for(int i=x1; i<(x1 + len); i++) 
				draw(i, y1 + dy * (i - x1) / dx, yuv);
		}
	} else {
		if(y1 < h/2) {
			for(int i=y1; i>(y1 - len); i--)
				draw(x1 + dx * (i - y1) / dy, i, yuv);
		} else {
			for(int i=y1; i<(y1 + len); i++)
				draw(x1 + dx * (i - y1) / dy, i, yuv);
		}
	}
}

__device__ void erase(int x1, int y1, uint8_t *yuv) {
	if(x1>=0 && x1<w && y1>=0 && y1<h)
		yuv[x1+y1*w] = 0;
}

__global__ void eraseStripe(int *x, int *y, int time, uint8_t *yuv, int len) {
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if(idx > S) return;
	int x1 = x[idx]*4;
	int y1 = y[idx]*4;
	int dx = x1 - w/2;
	int dy = y1 - h/2;
	if(abs(dx) > abs(dy)) {
		if(x1 < w/2) {
			for(int i=x1-len; i>=0; i--)
				draw(i, y1 + dy * (i - x1) / dx, yuv);
		} else {
			for(int i=x1+len; i<w; i++) 
				draw(i, y1 + dy * (i - x1) / dx, yuv);
		}
	} else {
		if(y1 < h/2) {
			for(int i=y1-len; i>=0; i--)
				draw(x1 + dx * (i - y1) / dy, i, yuv);
		} else {
			for(int i=y1+len; i<h; i++)
				draw(x1 + dx * (i - y1) / dy, i, yuv);
		}
	}
}

__global__ void drawCircle(uint8_t *yuv, int r, int color) {
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	int x = idx % w;
	int y = idx / w;
	int d = (x - w/2) * (x - w/2) + (y - h/2) * (y - h/2);
	if(d < r*r)
		yuv[idx] = color;
}

void Lab2VideoGenerator::Generate(uint8_t *yuv) {
	int *x;
	int *y;
	if(flag == 0) {
		printf("generate star\n");
		flag = 1;
		generate_star();
	}

// copy coordinate
	cudaMalloc(&x, S*sizeof(int));
	cudaMemcpy(x, tempx, S*sizeof(int), cudaMemcpyHostToDevice);
	cudaMalloc(&y, S*sizeof(int));
	cudaMemcpy(y, tempy, S*sizeof(int), cudaMemcpyHostToDevice);
	if(impl->t == 0) {
		test<<<1, 1>>>(x, y);
		printf("test\n");
	}
// set background to black
	cudaMemset(yuv, 0, W*H);
	//setPixelY<<<640, 480>>>(impl->t, yuv);
	
	if(impl->t < fps*5)
		paintStar<<<S/3+1, 3>>>(x, y, impl->t, yuv);
	else if(impl->t < fps*8){
		printStripe<<<S/3+1, 3>>>(x, y, impl->t, yuv, impl->t-fps*5);
	} else if(impl->t < fps*10) {
		printStripe<<<S/3+1, 3>>>(x, y, impl->t, yuv, 100 + (impl->t-fps*8)*10);
	} else if(impl->t < fps*13) {
		eraseStripe<<<S/3+1, 3>>>(x, y, impl->t, yuv, (impl->t-fps*10)*8);
	}

	cudaMemset(yuv+W*H, 128, W*H/2);
	
	if(impl->t > fps*10) {
		drawCircle<<<640, 480>>>(yuv, (impl->t-fps*10)*15, 255);
	}

	if(impl->t > fps*13.5) {
		drawCircle<<<640, 480>>>(yuv, (impl->t-fps*13.5)*15, 0);
		paintStar<<<S/3+1, 3>>>(x, y, impl->t, yuv);
	}

	cudaDeviceSynchronize();
	cudaMemset(yuv+W*H, 128, W*H/2);
	++(impl->t);
}








