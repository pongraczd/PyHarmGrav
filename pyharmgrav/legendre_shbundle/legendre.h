#pragma once

#include <stdio.h> // <-- wahrscheinlich nur unter Linux notwendig
#include <stdlib.h>

#include <math.h>  // <-- absolutely necessary!!
#ifdef MATLAB_MEX_FILE
  #include "mex.h" // <-- necessary in Matlab that "printf" works!
#endif

/////////////////////////////////////
// Nico's plm.m compatible functions
void init_legendre_1st_kind_plm(int lmax, int order, double *Wmm, double *Wlm_1, double *Wlm_2);

int legendre_1st_kind_plm(int lmax, int order, int numlp, int degreelen, double *degree,
			  int thetalen, double *theta, 
			  double *nL, double *dnL, double *ddnL, double *Wmm, double *Wlm_1, double *Wlm_2);

int legendre_1st_kind_xnum_plm(int lmax, int order, int numlp, int degreelen, double *degree,
			       int thetalen, double *theta, 
			       double *nL, double *dnL, double *ddnL, double *Wmm, double *Wlm_1, double *Wlm_2);

/////////////////////////////////////
// speed optimized functions

// preparational computations for Legendre functions in the order: P00, P10, P20, P30, ... P11, P12, P13, ... P22, P23, ...
void init_legendre_1st_kind_od_speed(int lmax, double *Wmm, double *Wlm_1, double *Wlm_2);

// computations for Legendre functions in the order: P00, P10, P20, P30, ... P11, P12, P13, ... P22, P23, ...
// HINT: put *dnL = *ddnL = NULL or only *ddnL = NULL if you dont need them
int legendre_1st_kind_od_speed(int lmax, int thetalen, double *theta,
        double *nL, double *dnL, double *ddnL, double *Wmm, double *Wlm_1, double *Wlm_2);
// like above
int legendre_1st_kind_od_xnum_speed(int lmax, int thetalen, double *theta,
        double *nL, double *dnL, double *ddnL, double *Wmm, double *Wlm_1, double *Wlm_2);

