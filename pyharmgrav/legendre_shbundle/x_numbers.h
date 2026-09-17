#pragma once

#include <math.h>  // <-- absolutely necessary!!

// X-numbers
// #define XNUMDEG 1500  // the function switches to X-numbers for Lmax >= XNUMDEG 
#define BIG     512L // 768
#define BIGS    (BIG / 2L) // BIGS = BIG / 2

struct xnumber {
  double x;
  int    i;
};

// normalises an xnumber
struct xnumber x_norm(struct xnumber X);

// normalise and save to output
double x_x2dbl(struct xnumber X) ;

// factor multiplication xnumbers addition: Z = alpha * X + beta * Y
struct xnumber x_multadd(double alpha, struct xnumber X, double beta, struct xnumber Y);

// factor with xnumbers multiplication: Z = alpha * X
struct xnumber x_mult(double alpha, struct xnumber X);

// x-numbers subtraction: Z = X - Y
struct xnumber x_sub(struct xnumber X, struct xnumber Y);

// x-numbers init: Z = (x, i)
struct xnumber x_init(double x, int i);
