#include <math.h>

// Normally distributed random numbers with standard deviation of 2.5,
// quantized to integer values and saturated to the range -7 to +7, then scaled
// by 16 (i.e. -112 to +112), and finally stored as signed chars.
void random_complex(ComplexInput* random_num, int length) {
  double u1,u2,r,theta,a,b;
  double stddev=2.5;
  for(int i=0; i<length; i++){
    u1 = (rand() / (double)(RAND_MAX));
    u2 = (rand() / (double)(RAND_MAX));
    if(u1==0.0) u1=0.5/RAND_MAX;
    if(u2==0.0) u2=0.5/RAND_MAX;
    // Do Box-Muller transform
    r = stddev * sqrt(-2.0*log(u1));
    theta = 2*M_PI*u2;
    a = r * cos(theta);
    b = r * sin(theta);
    // Quantize (TODO: unbiased rounding?)
    a = round(a);
    b = round(b);
    // Saturate
    if(a >  7.0) a =  7.0;
    if(a < -7.0) a = -7.0;
    if(b >  7.0) b =  7.0;
    if(b < -7.0) b = -7.0;

    // Simulate 4 bit data that has been multipled by 16 (via left shift by 4;
    // could multiply by 18 to maximize range, but that might be more expensive
    // than left shift by 4).
    // (i.e. {-112, -96, -80, ..., +80, +96, +112})
    random_num[i] = ComplexInput( ((int)a) << 4, ((int)b) << 4 );

    // Uncomment next line to simulate all zeros for every input.
    //random_num[i] = ComplexInput(0,0);
  }
}

void reorderMatrix(Complex *matrix) {

#if MATRIX_ORDER == REGISTER_TILE_TRIANGULAR_ORDER
  // reorder the matrix from REGISTER_TILE_TRIANGULAR_ORDER to TRIANGULAR_ORDER

  size_t matLength = NFREQUENCY * ((NSTATION/2+1)*(NSTATION/4)*NPOL*NPOL*4) * (NPULSAR + 1);
  Complex *tmp = new Complex[matLength];
  memset(tmp, '0', matLength);

  for(int f=0; f<NFREQUENCY; f++) {
    for(int i=0; i<NSTATION/2; i++) {
      for (int rx=0; rx<2; rx++) {
	for (int j=0; j<=i; j++) {
	  for (int ry=0; ry<2; ry++) {
	    int k = f*(NSTATION+1)*(NSTATION/2) + (2*i+rx)*(2*i+rx+1)/2 + 2*j+ry;
	    int l = f*4*(NSTATION/2+1)*(NSTATION/4) + (2*ry+rx)*(NSTATION/2+1)*(NSTATION/4) + i*(i+1)/2 + j;
	    for (int pol1=0; pol1<NPOL; pol1++) {
	      for (int pol2=0; pol2<NPOL; pol2++) {
		size_t tri_index = (k*NPOL+pol1)*NPOL+pol2;
		size_t reg_index = (l*NPOL+pol1)*NPOL+pol2;
		tmp[tri_index] = 
		  Complex(((int*)matrix)[reg_index], ((int*)matrix)[reg_index+matLength]);
	      }
	    }
	  }
	}
      }
    }
  }
   
  memcpy(matrix, tmp, matLength*sizeof(Complex));

  delete []tmp;

#elif MATRIX_ORDER == REAL_IMAG_TRIANGULAR_ORDER
  // reorder the matrix from REAL_IMAG_TRIANGULAR_ORDER to TRIANGULAR_ORDER
  
  size_t matLength = NFREQUENCY * ((NSTATION+1)*(NSTATION/2)*NPOL*NPOL) * (NPULSAR + 1);
  Complex *tmp = new Complex[matLength];

  for(int f=0; f<NFREQUENCY; f++){
    for(int i=0; i<NSTATION; i++){
      for (int j=0; j<=i; j++) {
	int k = f*(NSTATION+1)*(NSTATION/2) + i*(i+1)/2 + j;
        for (int pol1=0; pol1<NPOL; pol1++) {
	  for (int pol2=0; pol2<NPOL; pol2++) {
	    size_t index = (k*NPOL+pol1)*NPOL+pol2;
	    tmp[index] = Complex(((int*)matrix)[index], ((int*)matrix)[index+matLength]);
	  }
	}
      }
    }
  }

  memcpy(matrix, tmp, matLength*sizeof(Complex));

  delete []tmp;
#endif

  return;
}

//check that GPU calculation matches the CPU
//
// verbose=0 means just print summary.
// verbsoe=1 means print each differing basline/channel.
// verbose=2 and array_h!=0 means print each differing baseline and each input
//           sample that contributed to it.
void checkResult(Complex *gpu, Complex *cpu, int verbose=0, ComplexInput *array_h=0) {

  printf("Checking result...\n"); fflush(stdout);

  int errorCount=0;
  Complex error = Complex(0,0);
  Complex maxError = Complex(0,0);

  for(int i=0; i<NSTATION; i++){
    for (int j=0; j<=i; j++) {
      for (int pol1=0; pol1<NPOL; pol1++) {
	for (int pol2=0; pol2<NPOL; pol2++) {
	  for(int f=0; f<NFREQUENCY; f++){
	    int k = f*(NSTATION+1)*(NSTATION/2) + i*(i+1)/2 + j;
	    int index = (k*NPOL+pol1)*NPOL+pol2;
	    if(real(cpu[index]) != real(gpu[index]) || imag(cpu[index]) != imag(gpu[index])) {
	      error = cpu[index] - gpu[index];
	      if(abs(error) > abs(maxError)) {
	        maxError = error;
	      }
              if(verbose > 0) {
                printf("%d %d %d %d %d %d %d %d  %d  %d  %d\n", f, i, j, k, pol1, pol2, index,
                       real(cpu[index]), real(gpu[index]), imag(cpu[index]), imag(gpu[index]));
                if(verbose > 1 && array_h) {
                  Complex sum(0,0);
                  for(int t=0; t<NTIME; t++) {
                    ComplexInput in0 = array_h[t*NFREQUENCY*NSTATION*2 + f*NSTATION*2 + i*2 + pol1];
                    ComplexInput in1 = array_h[t*NFREQUENCY*NSTATION*2 + f*NSTATION*2 + j*2 + pol2];
                    Complex prod = convert(in0) * conj(convert(in1));
                    sum += prod;
                    printf(" %4d (%4d,%4d) (%4d,%4d) -> (%6d, %6d)\n", t,
                        real(in0), imag(in0),
                        real(in1), imag(in1),
                        real(prod), imag(prod));
                  }
                  printf("                                 (%6d, %6d)\n", real(sum), imag(sum));
                }
              }
	      errorCount++;
	    }
	  }
	}
      }
    }
  }

  if (errorCount) {
    printf("Outer product summation failed with %d deviations [max error (%d, %d)]\n\n", errorCount, real(maxError), imag(maxError));
  } else {
    printf("Outer product summation successful [max error [%d, %d])\n\n", real(maxError), imag(maxError));
  }

}

// Extracts the full matrix from the packed Hermitian form
void extractMatrix(Complex *matrix, Complex *packed) {

  for(int f=0; f<NFREQUENCY; f++){
    for(int i=0; i<NSTATION; i++){
      for (int j=0; j<=i; j++) {
	int k = f*(NSTATION+1)*(NSTATION/2) + i*(i+1)/2 + j;
        for (int pol1=0; pol1<NPOL; pol1++) {
	  for (int pol2=0; pol2<NPOL; pol2++) {
	    int index = (k*NPOL+pol1)*NPOL+pol2;
	    matrix[(((f*NSTATION + i)*NSTATION + j)*NPOL + pol1)*NPOL+pol2] = 
	      packed[index];
	    matrix[(((f*NSTATION + j)*NSTATION + i)*NPOL + pol2)*NPOL+pol1] = conj(packed[index]);
	    //printf("%d %d %d %d %d %d %d\n",f,i,j,k,pol1,pol2,index);
	  }
	}
      }
    }
  }

}
