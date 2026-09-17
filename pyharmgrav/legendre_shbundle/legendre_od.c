#include <stdio.h>
#include <stdlib.h>

#include "legendre.h"
#include "x_numbers.h"

// preparational computations for Legendre functions in the order: P00, P10, P20, P30, ... P11, P21, P31, ... P22, P32, ...
void init_legendre_1st_kind_od_speed(int lmax, double *Wmm, double *Wlm_1, double *Wlm_2) {
  int l, m;
  int id_mm = 0, id_lm_1 = 0, id_lm_2 = 0;
  
  for (m = 0; m <= lmax; m++) {
    if (m != 0) {
      if (m == 1) {
        Wmm[id_mm++] = sqrt(3.0);
      } else {
        Wmm[id_mm++] = sqrt((2.0 * m + 1.0) / (2.0 * m));
      }
    }
    if ((m + 1) <= lmax) {
      Wlm_1[id_lm_1++] = sqrt(2.0 * m + 3.0);
    }
    for (l = m + 2; l <= lmax; l++) {
      Wlm_1[id_lm_1++] = sqrt((4.0 * l * l - 1.0) / ((l + m) * (l - m)));
      Wlm_2[id_lm_2++] = sqrt(1.0 - (4.0 * m * m - 1.0) / ((l + m) * (l - m) * (2.0 * l - 3.0)));
    }
  }
}



// Legendre functions in the order: P00, P10, P20, P30, ... P11, P21, P31, ... P22, P32, ...
// only possible combinations for passing output variables:
//   - nL != NULL
//   - nL, ndL != NULL
//   - nL, ndL, nddL != NULL
// ATTENTION! The user must make sure that size of nL, ndL, nddL fits to the length of theta and lmax!
int legendre_1st_kind_od_speed(int lmax, 
        int thetalen, double *theta,
        double *nL, double *dnL, double *ddnL,
        double *Wmm, double *Wlm_1, double *Wlm_2) {
  int l, m, t;
  int id_x = 0, id_mm, id_lm_1, id_lm_2;
  
  double tmp, tmp0, tmp1, tmp2,
    dtmp, dtmp0, dtmp1, dtmp2,
    ddtmp, ddtmp0, ddtmp1, ddtmp2;
  double ct, st;
  double iWmm, iWlm_1, iWlm_2;
  
  if ((!theta) || (!Wmm) || (!Wlm_1) || (!Wlm_2)) {
    printf("TAIHEN: variable is NULL!! (theta, Wmm, Wlm_1 or Wlm_2)\n");
    return 0;
  }

  id_x = 0;
  for (t = 0; t < thetalen; t++) {
    id_mm = id_lm_1 = id_lm_2 = 0;

    ct = cos(theta[t]);
    st = sin(theta[t]);
    
    if ((nL) && (!dnL) && (!ddnL)) { // nL only
      for (m = 0; m <= lmax; m++) {
        if (m == 0) // we're on the diagonal (first element): m = l = 0
          nL[id_x++] = tmp = tmp0 = 1.0;
        else // we're on the diagonal : m = l
          nL[id_x++] = tmp = tmp0 = Wmm[id_mm++] * st * tmp;
        
        if ((m + 1) <= lmax) // we're on the second element (next to the diagonal element)
          nL[id_x++] = tmp1 = Wlm_1[id_lm_1++] * ct * tmp0;
        
        for (l = m + 2; l <= lmax; l++) { // we're calculating the rest of the line
          nL[id_x++] = tmp2 = Wlm_1[id_lm_1++] * ct * tmp1 - Wlm_2[id_lm_2++] * tmp0;
          tmp0 = tmp1; tmp1 = tmp2;
        }
      }
    } else if ((nL) && (dnL) && (!ddnL)) { // nL and dnL only
      for (m = 0; m <= lmax; m++) {
        if (m == 0) { // we're on the diagonal (first element): m = l = 0
          dnL[id_x]  = dtmp = dtmp0 = 0.0;
          nL[id_x]   = tmp  = tmp0  = 1.0;
	  id_x++;
        } else { // we're on the diagonal : m = l
	  iWmm = Wmm[id_mm++];
          dnL[id_x]  = dtmp = dtmp0 = iWmm * (st * dtmp + ct * tmp);
          nL[id_x] = tmp  = tmp0  = iWmm * st * tmp;
	  id_x++;
        }
        
        if ((m + 1) <= lmax) { // we're on the second element (next to the diagonal element)
	  iWlm_1 = Wlm_1[id_lm_1++];
          dnL[id_x]  = dtmp1 = iWlm_1 * (ct * dtmp0 - st * tmp0); 
          nL[id_x]   = tmp1  = iWlm_1 * ct * tmp0;
	  id_x++;
        }
        
        for (l = m + 2; l <= lmax; l++) { // we're calculating the rest of the line
	  iWlm_1 = Wlm_1[id_lm_1++]; iWlm_2 = Wlm_2[id_lm_2++];
	  dnL[id_x]  = dtmp2 = iWlm_1 * (ct * dtmp1 - st * tmp1) - iWlm_2 * dtmp0;
          nL[id_x]   = tmp2 = iWlm_1 * ct * tmp1 - iWlm_2 * tmp0;
	  id_x++;
          dtmp0 = dtmp1; dtmp1 = dtmp2;
          tmp0  = tmp1;  tmp1  = tmp2;
        }
      }
    } else if ((nL) && (dnL) && (ddnL)) { // nL, dnL and ddnL
      for (m = 0; m <= lmax; m++) {
        if (m == 0) { // we're on the diagonal (first element): m = l = 0
          ddnL[id_x] = ddtmp = ddtmp0 = 0.0;
          dnL[id_x]  = dtmp  = dtmp0 = 0.0;
          nL[id_x]   = tmp   = tmp0  = 1.0;
	  id_x++;
        } else { // we're on the diagonal : m = l
	  iWmm = Wmm[id_mm++];
          ddnL[id_x] = ddtmp = ddtmp0 = iWmm * (st * (ddtmp - tmp) + 2.0 * ct * dtmp);	
          dnL[id_x]  = dtmp  = dtmp0  = iWmm * (st * dtmp + ct * tmp);
          nL[id_x]   = tmp   = tmp0   = iWmm * st * tmp;
	  id_x++;
        }
              
        if ((m + 1) <= lmax) { // we're on the second element (next to the diagonal element)
	  iWlm_1 = Wlm_1[id_lm_1++];
          ddnL[id_x] = ddtmp1 = iWlm_1 * (ct * (ddtmp0 - tmp0) - 2.0 * st * dtmp0); 
          dnL[id_x]  = dtmp1  = iWlm_1 * (ct * dtmp0 - st * tmp0); 
          nL[id_x]   = tmp1   = iWlm_1 * ct * tmp0;
	  id_x++;
        }
        
        for (l = m + 2; l <= lmax; l++) { // we're calculating the rest of the line
	  iWlm_1 = Wlm_1[id_lm_1++]; iWlm_2 = Wlm_2[id_lm_2++];
          ddnL[id_x] = ddtmp2 = iWlm_1 * (ct * (ddtmp1 - tmp1) - 2.0 * st * dtmp1) - iWlm_2 * ddtmp0;
          dnL[id_x]  = dtmp2  = iWlm_1 * (ct * dtmp1 - st * tmp1) - iWlm_2 * dtmp0;
          nL[id_x]   = tmp2   = iWlm_1 * ct * tmp1 - iWlm_2 * tmp0;
	  id_x++;
          ddtmp0 = ddtmp1; ddtmp1 = ddtmp2;
          dtmp0  = dtmp1;  dtmp1  = dtmp2;
          tmp0   = tmp1;   tmp1   = tmp2;
        }
      }
    } else { 
      printf("TAIHEN: Chosen combination of output variables nL, dnL, ddnL is not allowed!\n");
      return 0;
    }
  }
  return 1; // everything is fine
}



// X-number stabilized Legendre functions in the order: P00, P10, P20, P30, ... P11, P21, P31, ... P22, P32, ...
// only possible combinations for passing output variables:
//   - nL != NULL
//   - nL, ndL != NULL
//   - nL, ndL, nddL != NULL
// ATTENTION! The user must make sure that size of nL, ndL, nddL fits to the length of theta and lmax!
int legendre_1st_kind_od_xnum_speed(int lmax, 
        int thetalen, double *theta,
        double *nL, double *dnL, double *ddnL,
        double *Wmm, double *Wlm_1, double *Wlm_2) {
  int l, m, t;
  int id_x = 0, id_mm, id_lm_1, id_lm_2;
  
  struct xnumber x, x0, x1, x2,
    dx, dx0, dx1, dx2,
    ddx, ddx0, ddx1, ddx2;
  double ct, st;
  double iWmm, iWlm_1, iWlm_2;
  
  if ((!theta) || (!Wmm) || (!Wlm_1) || (!Wlm_2)) {
    printf("TAIHEN: variable is NULL!!\n");
    return 0;
  }

  id_x = 0;
  for (t = 0; t < thetalen; t++) {
    id_mm = id_lm_1 = id_lm_2 = 0;

    ct = cos(theta[t]);
    st = sin(theta[t]);
    
    if ((nL) && (!dnL) && (!ddnL)) { // nL only
      for (m = 0; m <= lmax; m++) {
        if (m == 0) { // we're on the diagonal (first element): m = l = 0
	  x0 = x = x_init(1.0, 0); nL[id_x++] = x_x2dbl(x0);
	} else { // we're on the diagonal : m = l
	  x0 = x = x_norm(x_mult(Wmm[id_mm++] * st, x)); nL[id_x++] = x_x2dbl(x0);
	}
              
        if ((m + 1) <= lmax) { // we're on the second element (next to the diagonal element)
	  x1 = x_norm(x_mult(Wlm_1[id_lm_1++] * ct, x0)); nL[id_x++] = x_x2dbl(x1);
	}
        
        for (l = m + 2; l <= lmax; l++) { // we're calculating the rest of the line
	  x2 = x_norm(x_multadd(Wlm_1[id_lm_1++] * ct, x1, -Wlm_2[id_lm_2++], x0)); nL[id_x++] = x_x2dbl(x2);   
	  x0 = x1; x1 = x2;
        }
      }
    } else if ((nL) && (dnL) && (!ddnL)) { // nL and dnL only
      for (m = 0; m <= lmax; m++) {
        if (m == 0) { // we're on the diagonal (first element): m = l = 0
	  dx0 = dx = x_init(0.0, 0); dnL[id_x] = x_x2dbl(dx0);
	  x0  = x  = x_init(1.0, 0); nL[id_x]  = x_x2dbl(x0);
	  id_x++;
        } else { // we're on the diagonal : m = l
	  iWmm = Wmm[id_mm++];
	  dx0 = dx = x_norm(x_mult(iWmm, x_multadd(st, dx, ct, x))); dnL[id_x] = x_x2dbl(dx0);
	  x0  = x  = x_norm(x_mult(iWmm * st, x));                   nL[id_x]  = x_x2dbl(x0);
	  id_x++;
        }
              
        if ((m + 1) <= lmax) { // we're on the second element (next to the diagonal element)
	  iWlm_1 = Wlm_1[id_lm_1++];
	  dx1 = x_norm(x_mult(iWlm_1, x_multadd(ct, dx0, -st, x0))); dnL[id_x] = x_x2dbl(dx1);
	  x1  = x_norm(x_mult(iWlm_1 * ct, x0));                     nL[id_x]  = x_x2dbl(x1);
	  id_x++;
        }
        
        for (l = m + 2; l <= lmax; l++) { // we're calculating the rest of the line
	  iWlm_1 = Wlm_1[id_lm_1++]; iWlm_2 = Wlm_2[id_lm_2++];
	  dx2 = x_norm(x_multadd(iWlm_1, x_multadd(ct, dx1, -st, x1), -iWlm_2, dx0)); dnL[id_x] = x_x2dbl(dx2);
	  x2  = x_norm(x_multadd(iWlm_1 * ct, x1, -iWlm_2, x0));                      nL[id_x]  = x_x2dbl(x2);   
	  id_x++;
          dx0 = dx1; dx1 = dx2;
	  x0 = x1; x1 = x2;
        }
      }
    } else if ((nL) && (dnL) && (ddnL)) { // nL, dnL and ddnL
      for (m = 0; m <= lmax; m++) {
        if (m == 0) { // we're on the diagonal (first element): m = l = 0
	  ddx0 = ddx = x_init(0.0, 0); ddnL[id_x] = x_x2dbl(ddx0); 
	  dx0  = dx  = x_init(0.0, 0); dnL[id_x]  = x_x2dbl(dx0);
	  x0   = x   = x_init(1.0, 0); nL[id_x]   = x_x2dbl(x0);
	  id_x++;
        } else { // we're on the diagonal : m = l
	  iWmm = Wmm[id_mm++];
	  ddx0 = ddx = x_norm(x_mult(iWmm, x_multadd(st, x_sub(ddx, x), 2.0 * ct, dx))); ddnL[id_x] = x_x2dbl(ddx0);
	  dx0  = dx  = x_norm(x_mult(iWmm, x_multadd(st, dx, ct, x)));                   dnL[id_x]  = x_x2dbl(dx0);
	  x0   = x   = x_norm(x_mult(iWmm * st, x));                                     nL[id_x]   = x_x2dbl(x0);
	  id_x++;
        }
              
        if ((m + 1) <= lmax) { // we're on the second element (next to the diagonal element)
	  iWlm_1 = Wlm_1[id_lm_1++];
	  ddx1 = x_norm(x_mult(iWlm_1, x_multadd(ct, x_sub(ddx0, x0), -2.0 * st, dx0))); ddnL[id_x] = x_x2dbl(ddx1);
	  dx1  = x_norm(x_mult(iWlm_1, x_multadd(ct, dx0, -st, x0)));                    dnL[id_x]  = x_x2dbl(dx1);
	  x1   = x_norm(x_mult(iWlm_1 * ct, x0));                                        nL[id_x]   = x_x2dbl(x1);
	  id_x++;
        }
        
        for (l = m + 2; l <= lmax; l++) { // we're calculating the rest of the line
	  iWlm_1 = Wlm_1[id_lm_1++]; iWlm_2 = Wlm_2[id_lm_2++];
	  ddx2 = x_norm(x_multadd(iWlm_1, x_multadd(ct, x_sub(ddx1, x1), -2.0 * st, dx1), -iWlm_2, ddx0)); ddnL[id_x] = x_x2dbl(ddx2);
	  dx2  = x_norm(x_multadd(iWlm_1, x_multadd(ct, dx1, -st, x1), -iWlm_2, dx0));   dnL[id_x]  = x_x2dbl(dx2);
	  x2   = x_norm(x_multadd(iWlm_1 * ct, x1, -iWlm_2, x0));                        nL[id_x]   = x_x2dbl(x2);   
	  id_x++;
          ddx0 = ddx1; ddx1 = ddx2;
          dx0 = dx1; dx1 = dx2;
	  x0 = x1; x1 = x2;
        }
      }
    } else {
      printf("TAIHEN: Chosen combination of output variables nL, dnL, ddnL is not allowed!\n");
      return 0;
    }
  }
  return 1; // everything is fine
}
