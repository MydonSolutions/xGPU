#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <complex>
#include <limits.h>
#include <omp.h>

/*
  Data ordering for input vectors is (running from slowest to fastest)
  [time][channel][station][polarization][complexity]

  Output matrix has ordering
  [channel][station][station][polarization][polarization][complexity]
*/

#define USE_GPU

// set the data type accordingly
typedef std::complex<char> ComplexInput;
#define COMPLEX_INPUT char2 
#define SCALE 1 // no need to rescale result 

#define TRIANGULAR_ORDER 1000
#define REAL_IMAG_TRIANGULAR_ORDER 2000
#define REGISTER_TILE_TRIANGULAR_ORDER 3000
#define MATRIX_ORDER REGISTER_TILE_TRIANGULAR_ORDER

// size = freq * time * station * pol *sizeof(ComplexInput)
#define GBYTE (1024llu*1024llu*1024llu)

#define NPOL 2
#define NSTATION 256ll
#define SIGNAL_SIZE GBYTE
#define SAMPLES SIGNAL_SIZE / (NSTATION*NPOL*sizeof(ComplexInput))
#define NFREQUENCY 10ll
#define NTIME 1000ll //SAMPLES / NFREQUENCY
#define NBASELINE ((NSTATION+1)*(NSTATION/2))
#define NDIM 2

//#define PIPE_LENGTH 1
//#define NTIME_PIPE NTIME / PIPE_LENGTH

#define NTIME_PIPE 100
#define PIPE_LENGTH NTIME / NTIME_PIPE

// how many pulsars are we binning for (Not implemented yet)
#define NPULSAR 0

// whether we are writing the matrix back to device memory (used for benchmarking)
int writeMatrix = 1;
// this must be enabled for this option to work though, slightly hurts performance
//#define WRITE_OPTION 

typedef std::complex<int> Complex;

Complex convert(const ComplexInput &b) {
  return Complex(real(b), imag(b));
}

// the OpenMP Xengine
#include "omp_xengine.cc"

// the GPU Xengine
#include "cuda_xengine.cu"

#include "cpu_util.cc"

int main(int argc, char** argv) {

  unsigned int seed = 1;
  int verbose = 0;

  if(argc>1) {
    seed = strtoul(argv[1], NULL, 0);
  }
  if(argc>2) {
    verbose = strtoul(argv[2], NULL, 0);
  }

  srand(seed);

  printf("Correlating %llu stations with %llu signals, with %llu channels and integration length %llu\n",
	 NSTATION, SAMPLES, NFREQUENCY, NTIME);

  unsigned long long vecLength = NFREQUENCY * NTIME * NSTATION * NPOL;


  // perform host memory allocation
  int packedMatLength = NFREQUENCY * ((NSTATION+1)*(NSTATION/2)*NPOL*NPOL);

  // allocate the GPU X-engine memory
  ComplexInput *array_h = 0; // this is pinned memory
  Complex *cuda_matrix_h = 0;
  xInit(&array_h, &cuda_matrix_h, NSTATION);

  // create an array of complex noise
  random_complex(array_h, vecLength);

  Complex *omp_matrix_h = (Complex *) malloc(packedMatLength*sizeof(Complex));
  printf("Calling CPU X-Engine\n");
#if (CUBE_MODE == CUBE_DEFAULT)
  ompXengine(omp_matrix_h, array_h);
#endif

  printf("Calling GPU X-Engine\n");
  cudaXengine(cuda_matrix_h, array_h);

#if (CUBE_MODE == CUBE_DEFAULT)
  
  reorderMatrix(cuda_matrix_h);
  checkResult(cuda_matrix_h, omp_matrix_h, verbose, array_h);

  int fullMatLength = NFREQUENCY * NSTATION*NSTATION*NPOL*NPOL;
  Complex *full_matrix_h = (Complex *) malloc(fullMatLength*sizeof(Complex));

  // convert from packed triangular to full matrix
  extractMatrix(full_matrix_h, cuda_matrix_h);

  free(full_matrix_h);
#endif

  //free host memory
  free(omp_matrix_h);

  // free gpu memory
  xFree(array_h, cuda_matrix_h);

  return 0;
}
