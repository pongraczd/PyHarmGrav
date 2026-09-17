#include <stdio.h>
#include <stdlib.h>

#include "legendre.h"
#include "x_numbers.h"

// preparational computations for Legendre functions in the order: P00, P10, P20, P30, ... P11, P21, P31, ... P22, P32, ...
void init_legendre_1st_kind_plm(int lmax, int msel, double *Wmm, double *Wlm_1, double *Wlm_2) {
  int l, m;
  int id_mm = 0, id_lm_1 = 0, id_lm_2 = 0;

  for (m = 0; m <= msel; m++) {
    if (m != 0) {
      if (m == 1) {
	Wmm[id_mm++] = sqrt(3.0); 
      } else {
	Wmm[id_mm++] = sqrt((2.0 * m + 1.0) / (2.0 * m)); 
      }
    }
    
    if (msel == m) { // in PLM-mode: calculate the whole line of l _only_ if msel = m
      if ((m + 1) <= lmax) {
	Wlm_1[id_lm_1++] = sqrt(2.0 * m + 3.0);
      }
      
      for (l = (m + 2); l <= lmax; l++) {
	Wlm_1[id_lm_1++] = sqrt((4.0 * l * l - 1.0) / ((l + m) * (l - m)));
	Wlm_2[id_lm_2++] = sqrt(1.0 - (4.0 * m * m - 1.0) / ((l + m) * (l - m) * (2.0 * l - 3.0)));
      }
    }
  }
}


// Legendre functions of order m
// only possible combinations for passing output variables:
//   - nL != NULL
//   - nL, ndL != NULL
//   - nL, ndL, nddL != NULL
// ATTENTION! The user must make sure that size of nL, ndL, nddL fits to the length of theta and lmax!
int legendre_1st_kind_plm(int lmax, int order, int numlp, int degreelen, double *degree,
			     int thetalen, double *theta,
			     double *nL, double *dnL, double *ddnL, double *Wmm, double *Wlm_1, double *Wlm_2) {
  int l, m, t;
  int id_x = 0, id_mm, id_lm_1, id_lm_2;
  double *L = NULL, *dL = NULL, *ddL = NULL;
  double ix0, ix1, ix2,
    idx0, idx1, idx2,
    iddx0, iddx1, iddx2;
  double ct, st;
  double iWmm, iWlm_1, iWlm_2;
  
  if ((!theta) || (!Wmm) || (!Wlm_1) || (!Wlm_2)) {
    printf("TAIHEN: variable is NULL!! (theta, Wmm, Wlm_1 or Wlm_2)\n");
    return 0;
  }

  // allocate memory
  if (nL)   L   = (double*)malloc(sizeof(double) * (numlp * (thetalen + 1)));
  if (dnL)  dL  = (double*)malloc(sizeof(double) * (numlp * (thetalen + 1)));
  if (ddnL) ddL = (double*)malloc(sizeof(double) * (numlp * (thetalen + 1)));

  if ((nL) && (!dnL) && (!ddnL)) { // nL only
    for (t = 0; t < thetalen; t++) {
      ct = cos(theta[t]);
      st = sin(theta[t]);
      
      id_mm = id_lm_1 = id_lm_2 = 0;
      id_x = t * numlp;
      
      // walk on the diagonal until desired order
      ix0 = 1.0; // we're on the diagonal (first element): m = l = 0
      for (m = 1; m <= order; m++) // we're on the diagonal : m = l
	ix0 = Wmm[id_mm++] * st * ix0;
      // after this for-loop, m is by one too big!!
      L[id_x++] = ix0; 
      
      // walk through the degrees of that line
      if (m <= lmax) // m contains already m + 1! - we're on the second element (next to the diagonal element)
	L[id_x++] = ix1 = Wlm_1[id_lm_1++] * ct * ix0;
      
      for (l = m + 1; l <= lmax; l++) { // m contains already m + 1! - we're calculating the rest of the line
	L[id_x++] = ix2 = Wlm_1[id_lm_1++] * ct * ix1 - Wlm_2[id_lm_2++] * ix0;
	ix0 = ix1; ix1 = ix2;
      } // for (l = m + 1; l <= lmax; l++)
    } // for (t = 0; t < thetalen; t++)
    // extract/collect the output
    id_x = 0; // initialize output index
    for (l = 0; l < degreelen; l++) {
      for (t = 0; t < thetalen; t++) {
	if (((int)degree[l] < order) || ((int)degree[l] < 0))
	  nL[id_x] = 0.0;
	else {
	  id_mm = t * numlp + (int)degree[l] - order; // calculate storage index
	  nL[id_x] = L[id_mm];      
	}
	id_x++; // increase output index
      }
    }
  } else if ((nL) && (dnL) && (!ddnL)) { // nL and dnL only
    for (t = 0; t < thetalen; t++) {  
      ct = cos(theta[t]);
      st = sin(theta[t]);
      
      id_mm = id_lm_1 = id_lm_2 = 0;
      id_x = t * numlp;
      
      // walk on the diagonal until desired order 
      idx0 = 0.0; // we're on the diagonal (first element): m = l = 0
      ix0  = 1.0;
      for (m = 1; m <= order; m++) { // we're on the diagonal : m = l
	iWmm = Wmm[id_mm++];
	idx0 = iWmm * (st * idx0 + ct * ix0);
	ix0  = iWmm * st * ix0;
      } // after this for-loop, m is by one too big!!
      dL[id_x] = idx0;
      L[id_x]  = ix0;
      id_x++;
      
      // walk through the degrees of that line
      if (m <= lmax) { // m contains alread m + 1! - we're on the second element (next to the diagonal element)
	iWlm_1 = Wlm_1[id_lm_1++];
	dL[id_x] = idx1 = iWlm_1 * (ct * idx0 - st * ix0); 
	L[id_x]  = ix1  = iWlm_1 * ct * ix0;
	id_x++;
      } // if ((m + 1) <= lmax)
      
      for (l = m + 1; l <= lmax; l++) { // m contains already m + 1! - we're calculating the rest of the line
	iWlm_1 = Wlm_1[id_lm_1++]; iWlm_2 = Wlm_2[id_lm_2++];
	dL[id_x] = idx2 = iWlm_1 * (ct * idx1 - st * ix1) - iWlm_2 * idx0;
	L[id_x]  = ix2  = iWlm_1 * ct * ix1               - iWlm_2 * ix0;
	id_x++;
	idx0 = idx1; idx1 = idx2;
	ix0 = ix1; ix1 = ix2;	
      } // for (l = m + 2; l <= lmax; l++)
    }
    id_x = 0; // initialize output index
    for (l = 0; l < degreelen; l++) {
      for (t = 0; t < thetalen; t++) {
	if (((int)degree[l] < order) || ((int)degree[l] < 0)) {
	  dnL[id_x] = nL[id_x] = 0.0;
	} else {
	  id_mm = t * numlp + (int)degree[l] - order; // calculate storage index
	  dnL[id_x] = dL[id_mm];
	  nL[id_x] = L[id_mm];      
	}
	id_x++; // increase output index
      }
    }
  } else if ((nL) && (dnL) && (ddnL)) { // nL, dnL and ddnL
    for (t = 0; t < thetalen; t++) {  
      ct = cos(theta[t]);
      st = sin(theta[t]);
      
      id_mm = id_lm_1 = id_lm_2 = 0;
      id_x = t * numlp;
      
      // walk on the diagonal until desired order 
      iddx0 = 0.0; // we're on the diagonal (first element): m = l = 0
      idx0  = 0.0;
      ix0   = 1.0;
      for (m = 1; m <= order; m++) { // we're on the diagonal : m = l
	iWmm = Wmm[id_mm++];
	iddx0 = iWmm * (st * (iddx0 - ix0) + 2.0 * ct * idx0);	
	idx0  = iWmm * (st * idx0 + ct * ix0);
	ix0   = iWmm * st * ix0;
      } // after this for-loop, m is by one too big!!
      ddL[id_x] = iddx0;
      dL[id_x]  = idx0;
      L[id_x]   = ix0;
      id_x++;
      
      // walk through the degrees of that line
      if (m <= lmax) { // m contains alread m + 1! - we're on the second element (next to the diagonal element)
	iWlm_1 = Wlm_1[id_lm_1++];
	ddL[id_x] = iddx1 = iWlm_1 * (ct * (iddx0 - ix0) - 2.0 * st * idx0); 
	dL[id_x]  = idx1  = iWlm_1 * (ct * idx0 - st * ix0);
	L[id_x]   = ix1   = iWlm_1 * ct * ix0;
	id_x++;
      } // if ((m + 1) <= lmax)
      
      for (l = m + 1; l <= lmax; l++) { // m contains alread m + 1! - we're calculating the rest of the line
	iWlm_1 = Wlm_1[id_lm_1++]; iWlm_2 = Wlm_2[id_lm_2++];
 	ddL[id_x] = iddx2 = iWlm_1 * (ct * (iddx1 - ix1) - 2.0 * st * idx1) - iWlm_2 * iddx0;
	dL[id_x]  = idx2  = iWlm_1 * (ct * idx1 - st * ix1)                 - iWlm_2 * idx0;
	L[id_x]   = ix2   = iWlm_1 * ct * ix1                               - iWlm_2 * ix0;
	id_x++;
	iddx0 = iddx1; iddx1 = iddx2;
	idx0 = idx1; idx1 = idx2;
	ix0 = ix1; ix1 = ix2;
      } // for (l = m + 2; l <= lmax; l++)
    }
    id_x = 0; // initialize output index
    for (l = 0; l < degreelen; l++) {
      for (t = 0; t < thetalen; t++) {
	if (((int)degree[l] < order) || ((int)degree[l] < 0)) {
	  ddnL[id_x] = dnL[id_x] = nL[id_x] = 0.0;
	} else {
	  id_mm = t * numlp + (int)degree[l] - order; // calculate storage index
	  ddnL[id_x] = ddL[id_mm];
	  dnL[id_x]  = dL[id_mm];
	  nL[id_x]   = L[id_mm];      
	}
	id_x++; // increase output index
      }
    }	
  } else { // unsupported combination of
    printf("TAIHEN: Chosen combination of output variables nL, dnL, ddnL is not allowed!\n");
    return 0;
  }

  // don't forget to free the memory again!
  if (nL)   free(L);
  if (dnL)  free(dL);
  if (ddnL) free(ddL);

  return 1; // everything is fine
}


// X-number stabilized Legendre functions of order m
// only possible combinations for passing output variables:
//   - nL != NULL
//   - nL, ndL != NULL
//   - nL, ndL, nddL != NULL
// ATTENTION! The user must make sure that size of nL, ndL, nddL fits to the length of theta and lmax!
int legendre_1st_kind_xnum_plm(int lmax, int order, int numlp, int degreelen, double *degree,
			       int thetalen, double *theta,
			       double *nL, double *dnL, double *ddnL, double *Wmm, double *Wlm_1, double *Wlm_2) {
  int l, m, t;
  int id_x, id_mm, id_lm_1, id_lm_2;
  double *L = NULL, *dL = NULL, *ddL = NULL;
  struct xnumber 
    x0, x1, x2,
    dx0, dx1, dx2,
    ddx0, ddx1, ddx2;
  double ct, st;
  double iWmm, iWlm_1, iWlm_2;
  
  if ((!theta) || (!Wmm) || (!Wlm_1) || (!Wlm_2)) {
    printf("TAIHEN: variable is NULL!! (theta, Wmm, Wlm_1 or Wlm_2)\n");
    return 0;
  }

  // allocate memory
  if (nL)   L   = (double*)malloc(sizeof(double) * (numlp * (thetalen + 1)));
  if (dnL)  dL  = (double*)malloc(sizeof(double) * (numlp * (thetalen + 1)));
  if (ddnL) ddL = (double*)malloc(sizeof(double) * (numlp * (thetalen + 1)));

  if ((nL) && (!dnL) && (!ddnL)) { // nL only
    for (t = 0; t < thetalen; t++) {
      ct = cos(theta[t]);
      st = sin(theta[t]);
      
      id_mm = id_lm_1 = id_lm_2 = 0;
      id_x = t * numlp;
      
      // walk on the diagonal until desired order
      x0 = x_init(1.0, 0); // we're on the diagonal (first element): m = l = 0
      for (m = 1; m <= order; m++) // we're on the diagonal: m = l 
	x0 = x_norm(x_mult(Wmm[id_mm++] * st, x0));
      // after this for-loop, m is by one too big!!
      L[id_x++] = x_x2dbl(x0); 

      if (m <= lmax) { // m contains already m + 1! - we're on the second element (next to the diagonal element)
	x1 = x_norm(x_mult(Wlm_1[id_lm_1++] * ct, x0));
	L[id_x++] = x_x2dbl(x1); 
      } // if (m <= lmax) 
	    
      for (l = (m + 1); l <= lmax; l++) { // m contains already m + 1! - we're calculating the rest of the line
	x2 = x_norm(x_multadd(Wlm_1[id_lm_1++] * ct, x1, -Wlm_2[id_lm_2++], x0));
	L[id_x++] = x_x2dbl(x2);
	x0 = x1; x1 = x2;
      } // for (l = (m + 1); l <= lmax ...
    } // for (t = 0; t < thetalen ...
    // extract/collect the output
    id_x = 0; // initialize output index
    for (l = 0; l < degreelen; l++) {
      for (t = 0; t < thetalen; t++) {
	if (((int)degree[l] < order) || ((int)degree[l] < 0))
	  nL[id_x] = 0.0;
	else {
	  id_mm = t * numlp + (int)degree[l] - order; // calculate storage index
	  nL[id_x] = L[id_mm];      
	}
	id_x++; // increase output index
      }
    }
  } else if ((nL) && (dnL) && (!ddnL)) { // nL and dnL only
    for (t = 0; t < thetalen; t++) {
      ct = cos(theta[t]);
      st = sin(theta[t]);
	
      id_mm = id_lm_1 = id_lm_2 = 0;
      id_x = t * numlp;
	
      // walk on the diagonal until desired order
      dx0 = x_init(0.0, 0); // we're on the diagonal (first element): m = l = 0
      x0  = x_init(1.0, 0);
      for (m = 1; m <= order; m++) { // we're on the diagonal : m = l
	iWmm = Wmm[id_mm++];
	dx0 = x_norm(x_mult(iWmm, x_multadd(st, dx0, ct, x0)));
	x0  = x_norm(x_mult(iWmm * st, x0));
      } // after this for-loop, m is by one too big!!
      dL[id_x] = x_x2dbl(dx0);
      L[id_x]  = x_x2dbl(x0);
      id_x++;
	  
      if (m <= lmax) { // m contains already m + 1! - we're on the second element (next to the diagonal element)
	iWlm_1 = Wlm_1[id_lm_1++];
 	dx1 = x_norm(x_mult(iWlm_1, x_multadd(ct, dx0, -st, x0)));
	x1  = x_norm(x_mult(iWlm_1 * ct, x0));
	dL[id_x] = x_x2dbl(dx1);
	L[id_x]  = x_x2dbl(x1); 
	id_x++;
      } // if (m <= lmax) 
	    
      for (l = (m + 1); l <= lmax; l++) { // we're calculating the rest of the line
	iWlm_1 = Wlm_1[id_lm_1++]; iWlm_2 = Wlm_2[id_lm_2++];
	dx2 = x_norm(x_multadd(iWlm_1, x_multadd(ct, dx1, -st, x1), -iWlm_2, dx0));
	x2  = x_norm(x_multadd(iWlm_1 * ct, x1, -iWlm_2, x0));
	dL[id_x] = x_x2dbl(dx2);
	L[id_x]  = x_x2dbl(x2);
	id_x++;
	dx0 = dx1; dx1 = dx2;
	x0 = x1; x1 = x2;
      } // for (l = (m + 2); l <= lmax ...
    } // for (t = 0; t < thetalen ...
    id_x = 0; // initialize output index
    for (l = 0; l < degreelen; l++) {
      for (t = 0; t < thetalen; t++) {
	if (((int)degree[l] < order) || ((int)degree[l] < 0)) {
	  dnL[id_x] = nL[id_x] = 0.0;
	} else {
	  id_mm = t * numlp + (int)degree[l] - order; // calculate storage index
	  dnL[id_x] = dL[id_mm];
	  nL[id_x] = L[id_mm];      
	}
	id_x++; // increase output index
      }
    }
  } else if ((nL) && (dnL) && (ddnL)) { // nL, dnL and ddnL
    for (t = 0; t < thetalen; t++) {
      ct = cos(theta[t]);
      st = sin(theta[t]);
      
      id_mm = id_lm_1 = id_lm_2 = 0;
      id_x = t * numlp;

      // walk on the diagonal until desired order 
      ddx0 = x_init(0.0, 0); // we're on the diagonal (first element): m = l = 0
      dx0  = x_init(0.0, 0);
      x0   = x_init(1.0, 0);
      for (m = 1; m <= order; m++) { // we're on the diagonal : m = l
	iWmm = Wmm[id_mm++];
	ddx0 = x_norm(x_mult(iWmm, x_multadd(st, x_sub(ddx0, x0), 2.0 * ct, dx0)));
	dx0  = x_norm(x_mult(iWmm, x_multadd(st, dx0, ct, x0)));
	x0   = x_norm(x_mult(iWmm * st, x0));
      } // after this for-loop, m is by one too big!!
      ddL[id_x] = x_x2dbl(ddx0);
      dL[id_x]  = x_x2dbl(dx0);
      L[id_x]   = x_x2dbl(x0);
      id_x++;
	  
      if (m <= lmax) { // m contains alread m + 1! - we're on the second element (next to the diagonal element)
	iWlm_1 = Wlm_1[id_lm_1++];
	ddx1 = x_norm(x_mult(iWlm_1, x_multadd(ct, x_sub(ddx0, x0), -2.0 * st, dx0)));
	dx1  = x_norm(x_mult(iWlm_1, x_multadd(ct, dx0, -st, x0)));
	x1   = x_norm(x_mult(iWlm_1 * ct, x0));
  	ddL[id_x] = x_x2dbl(ddx1);
	dL[id_x]  = x_x2dbl(dx1);
	L[id_x]   = x_x2dbl(x1);
	id_x++;
     } // if (m <= lmax) 
	    
      for (l = (m + 1); l <= lmax; l++) { // m contains alread m + 1! - we're calculating the rest of the line
	iWlm_1 = Wlm_1[id_lm_1++]; iWlm_2 = Wlm_2[id_lm_2++];
	ddx2 = x_norm(x_multadd(iWlm_1, x_multadd(ct, x_sub(ddx1, x1), -2.0 * st, dx1), -iWlm_2, ddx0));
	dx2  = x_norm(x_multadd(iWlm_1, x_multadd(ct, dx1, -st, x1), -iWlm_2, dx0));
	x2   = x_norm(x_multadd(iWlm_1 * ct, x1, -iWlm_2, x0));
	ddL[id_x] = x_x2dbl(ddx2);
	dL[id_x]  = x_x2dbl(dx2);
	L[id_x]   = x_x2dbl(x2);
	id_x++;
	ddx0 = ddx1; ddx1 = ddx2;
	dx0 = dx1; dx1 = dx2;
	x0 = x1; x1 = x2;
      } // for (l = (m + 1); l <= lmax ...
    } // for (t = 0; t < thetalen ...
    id_x = 0; // initialize output index
    for (l = 0; l < degreelen; l++) {
      for (t = 0; t < thetalen; t++) {
	if (((int)degree[l] < order) || ((int)degree[l] < 0)) {
	  ddnL[id_x] = dnL[id_x] = nL[id_x] = 0.0;
	} else {
	  id_mm = t * numlp + (int)degree[l] - order; // calculate storage index
	  ddnL[id_x] = ddL[id_mm];
	  dnL[id_x]  = dL[id_mm];
	  nL[id_x]   = L[id_mm];      
	}
	id_x++; // increase output index
      }
    }
  } else { // unsupported combination of
    printf("TAIHEN: Chosen combination of output variables nL, dnL, ddnL is not allowed!\n");
    return 0;
  }
  
  // don't forget to free the memory again!
  if (nL)   free(L);
  if (dnL)  free(dL);
  if (ddnL) free(ddL);
  
  return 1; // everything is fine
}
