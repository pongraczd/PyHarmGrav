#include "x_numbers.h"

// normalises an xnumber
struct xnumber x_norm(struct xnumber X)
{ 
  struct xnumber Z = X;
  double r;
  int ex;

  while (1) {
    r = frexp(Z.x, &ex);
    if (ex >= BIGS) {
      Z.x = ldexp(r, ex - BIG); Z.i++; 
    } else if (ex < -BIGS) {
      Z.x = ldexp(r, ex + BIG); Z.i--;
    } else // everything is ok
      break;
  }
  return Z;
}

// normalise and save to output
double x_x2dbl(struct xnumber X) 
{
  if ((X.i) == 0)
    return X.x; 
  else
    return ldexp(X.x, X.i * BIG); 
}

// factor multiplication xnumbers addition: Z = alpha * X + beta * Y
struct xnumber x_multadd(double alpha, struct xnumber X,
			 double beta, struct xnumber Y)
{
  struct xnumber Z;
  int id;
  
  id = X.i - Y.i;
  if (id == 0) {
    Z.x = alpha * X.x + beta * Y.x;
    Z.i = X.i;
  } else if (id >= 1) {
    Z.x = alpha * X.x + beta * ldexp(Y.x, -id * BIG);
    Z.i = X.i;
  } else { // if (id <= -1) 
    Z.x = alpha * ldexp(X.x, -id * BIG) + beta * Y.x;
    Z.i = Y.i;
  }
  return Z;
}

// factor with xnumbers multiplication: Z = alpha * X
struct xnumber x_mult(double alpha, struct xnumber X)
{
  struct xnumber Z;
  Z.x = alpha * X.x;
  Z.i = X.i;
  return Z;
}

// x-numbers subtraction: Z = X - Y
struct xnumber x_sub(struct xnumber X, struct xnumber Y)
{
  struct xnumber Z;
  int id = X.i - Y.i;

  if (id == 0) {
    Z.x = X.x - Y.x;
    Z.i = X.i;
  } else if (id >= 1) {
    Z.x = X.x - ldexp(Y.x, -id * BIG);
    Z.i = X.i;
  } else { // if (id <= -1)
    Z.x = ldexp(X.x, -id * BIG) - Y.x;
    Z.i = Y.i;
  }
  return Z; 
}

// x-numbers init: Z = (x, i)
struct xnumber x_init(double x, int i)
{
  struct xnumber Z;
  Z.x = x; Z.i = i;
  return Z;
}
