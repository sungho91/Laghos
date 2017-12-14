// Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-734707. All Rights
// reserved. See files LICENSE and NOTICE for details.
//
// This file is part of CEED, a collection of benchmarks, miniapps, software
// libraries and APIs for efficient high-order finite element and spectral
// element discretizations for exascale applications. For more information and
// source code availability see http://github.com/ceed.
//
// The CEED research is supported by the Exascale Computing Project 17-SC-20-SC,
// a collaborative effort of two U.S. Department of Energy organizations (Office
// of Science and the National Nuclear Security Administration) responsible for
// the planning and preparation of a capable exascale ecosystem, including
// software, applications, hardware, advanced system engineering and early
// testbed platforms, in support of the nation's exascale computing imperative.
#include "raja.hpp"

// *****************************************************************************
void rInitQuadratureData(const int NUM_QUAD,
                         const int numElements,
                         const double* rho0,
                         const double* detJ,
                         const double* quadWeights,
                         double* __restrict rho0DetJ0w) {
  forall(el,numElements,{
      for (int q = 0; q < NUM_QUAD; ++q){
        rho0DetJ0w[ijN(q,el,NUM_QUAD)] =
          rho0[ijN(q,el,NUM_QUAD)]*detJ[ijN(q,el,NUM_QUAD)]*quadWeights[q];
      }
    });
}

// *****************************************************************************
void rUpdateQuadratureData2D(const double GAMMA,
                             const double H0,
                             const double CFL,
                             const bool USE_VISCOSITY,
                             const int NUM_DIM,
                             const int NUM_QUAD,
                             const int NUM_QUAD_1D,
                             const int NUM_QUAD_2D,
                             const int NUM_DOFS_1D,
                             const int numElements,
                             const double* dofToQuad,
                             const double* dofToQuadD,
                             const double* quadWeights,
                             const double* v,
                             const double* e,
                             const double* rho0DetJ0w,
                             const double* invJ0,
                             const double* J,
                             const double* invJ,
                             const double* detJ,
                             double* __restrict stressJinvT,
                             double* __restrict dtEst) {
//  printf("\033[31m[%d]\033[m\n",NUM_DIM);
//  printf("\033[31m[%d]\033[m\n",NUM_QUAD_1D);
//  printf("\033[31m[%d]\033[m\n",NUM_QUAD_2D);
  assert(NUM_DIM==2); const int nd = 2;
  assert(NUM_QUAD_1D==4); const int q1 = 4;
  assert(NUM_QUAD_2D==16); const int q2 = 16;
  forall(el,numElements,{
    double s_gradv[4*q2] ;
    for (int i = 0; i < (4*NUM_QUAD_2D); ++i) {
      s_gradv[i] = 0;
    }

    for (int dy = 0; dy < NUM_DOFS_1D; ++dy) {
      double vDx[2*q1] ;
      double vx[2*q1]  ;
      for (int qx = 0; qx < NUM_QUAD_1D; ++qx) {
        for (int vi = 0; vi < 2; ++vi) {
          vDx[ijN(vi,qx,2)] = 0;
          vx[ijN(vi,qx,2)] = 0;
        }
      }

      for (int dx = 0; dx < NUM_DOFS_1D; ++dx) {
        for (int qx = 0; qx < NUM_QUAD_1D; ++qx) {
          for (int vi = 0; vi < 2; ++vi) {
            vDx[ijN(vi,qx,2)] += v[_ijklNM(vi,dx,dy,el,NUM_DOFS_1D,numElements)]*dofToQuadD[ijN(qx,dx,NUM_QUAD_1D)];
            vx[ijN(vi,qx,2)]  += v[_ijklNM(vi,dx,dy,el,NUM_DOFS_1D,numElements)]*dofToQuad[ijN(qx,dx,NUM_QUAD_1D)];
          }
        }
      }

      for (int qy = 0; qy < NUM_QUAD_1D; ++qy) {
        const double wy  = dofToQuad[ijN(qy,dy,NUM_QUAD_1D)];
        const double wDy = dofToQuadD[ijN(qy,dy,NUM_QUAD_1D)];

        for (int qx = 0; qx < NUM_QUAD_1D; ++qx) {
          for (int vi = 0; vi < 2; ++vi) {
            s_gradv[ijkN(vi,0,qx+qy*NUM_QUAD_1D,2)] += wy *vDx[ijN(vi,qx,2)];
            s_gradv[ijkN(vi,1,qx+qy*NUM_QUAD_1D,2)] += wDy*vx[ijN(vi,qx,2)];
          }
        }
      }
    }

    for (int q = 0; q < NUM_QUAD; ++q) {
      double q_gradv[nd*nd];
      double q_stress[nd*nd];

      const double invJ_00 = invJ[ijklNM(0,0,q,el,NUM_DIM,NUM_QUAD)];
      const double invJ_10 = invJ[ijklNM(1,0,q,el,NUM_DIM,NUM_QUAD)];
      const double invJ_01 = invJ[ijklNM(0,1,q,el,NUM_DIM,NUM_QUAD)];
      const double invJ_11 = invJ[ijklNM(1,1,q,el,NUM_DIM,NUM_QUAD)];

      q_gradv[ijN(0,0,2)] = ((s_gradv[ijkN(0,0,q,2)]*invJ_00)+(s_gradv[ijkN(1,0,q,2)]*invJ_01));
      q_gradv[ijN(1,0,2)] = ((s_gradv[ijkN(0,0,q,2)]*invJ_10)+(s_gradv[ijkN(1,0,q,2)]*invJ_11));
      q_gradv[ijN(0,1,2)] = ((s_gradv[ijkN(0,1,q,2)]*invJ_00)+(s_gradv[ijkN(1,1,q,2)]*invJ_01));
      q_gradv[ijN(1,1,2)] = ((s_gradv[ijkN(0,1,q,2)]*invJ_10)+(s_gradv[ijkN(1,1,q,2)]*invJ_11));

      const double q_Jw = detJ[ijN(q,el,NUM_QUAD)]*quadWeights[q];

      const double q_rho = rho0DetJ0w[ijN(q,el,NUM_QUAD)] / q_Jw;
      const double q_e   = fmax(0.0,e[ijN(q,el,NUM_QUAD)]);

      // TODO: Input OccaVector eos(q,e) -> (stress,soundSpeed)
      const double s = -(GAMMA-1.0)*q_rho*q_e;
      q_stress[ijN(0,0,2)] = s; q_stress[ijN(1,0,2)] = 0;
      q_stress[ijN(0,1,2)] = 0; q_stress[ijN(1,1,2)] = s;

      const double gradv00 = q_gradv[ijN(0,0,2)];
      const double gradv11 = q_gradv[ijN(1,1,2)];
      const double gradv10 = 0.5*(q_gradv[ijN(1,0,2)]+q_gradv[ijN(0,1,2)]);
      q_gradv[ijN(1,0,2)] = gradv10;
      q_gradv[ijN(0,1,2)] = gradv10;

      double comprDirX = 1;
      double comprDirY = 0;
      double minEig = 0;
      // linalg/densemat.cpp: Eigensystem2S()
      if (gradv10 == 0) {
        minEig = (gradv00 < gradv11) ? gradv00 : gradv11;
      } else {
        const double zeta  = (gradv11-gradv00) / (2.0*gradv10);
        const double azeta = fabs(zeta);
        double t = 1.0 / (azeta+sqrt(1.0+zeta*zeta));
        if ((t < 0) != (zeta < 0)) {
          t = -t;
        }
        const double c = sqrt(1.0 / (1.0+t*t));
        const double s = c*t;
        t *= gradv10;
        if ((gradv00-t) <= (gradv11+t)) {
          minEig = gradv00-t;
          comprDirX = c;
          comprDirY = -s;
        } else {
          minEig = gradv11+t;
          comprDirX = s;
          comprDirY = c;
        }
      }

      // Computes the initial->physical transformation Jacobian.
      const double J_00 = J[ijklNM(0,0,q,el,NUM_DIM,NUM_QUAD)];
      const double J_10 = J[ijklNM(1,0,q,el,NUM_DIM,NUM_QUAD)];
      const double J_01 = J[ijklNM(0,1,q,el,NUM_DIM,NUM_QUAD)];
      const double J_11 = J[ijklNM(1,1,q,el,NUM_DIM,NUM_QUAD)];
      const double invJ0_00 = invJ0[ijklNM(0,0,q,el,NUM_DIM,NUM_QUAD)];
      const double invJ0_10 = invJ0[ijklNM(1,0,q,el,NUM_DIM,NUM_QUAD)];
      const double invJ0_01 = invJ0[ijklNM(0,1,q,el,NUM_DIM,NUM_QUAD)];
      const double invJ0_11 = invJ0[ijklNM(1,1,q,el,NUM_DIM,NUM_QUAD)];
      const double Jpi_00 = ((J_00*invJ0_00)+(J_10*invJ0_01));
      const double Jpi_10 = ((J_00*invJ0_10)+(J_10*invJ0_11));
      const double Jpi_01 = ((J_01*invJ0_00)+(J_11*invJ0_01));
      const double Jpi_11 = ((J_01*invJ0_10)+(J_11*invJ0_11));
      const double physDirX = (Jpi_00*comprDirX)+(Jpi_10*comprDirY);
      const double physDirY = (Jpi_01*comprDirX)+(Jpi_11*comprDirY);
      const double q_h = H0*sqrt((physDirX*physDirX)+(physDirY*physDirY));
      // TODO: soundSpeed will be an input as well (function call or values per q)
      const double soundSpeed = sqrt(GAMMA*(GAMMA-1.0)*q_e);
      dtEst[ijN(q,el,NUM_QUAD)] = CFL*q_h / soundSpeed;
      //printf("\ndt_est=%.15e",q_h);
      //printf("\ndt_est=%.15e",dtEst[ijN(q,el)]);
      if (USE_VISCOSITY) {
        // TODO: Check how we can extract outside of kernel
        const double mu = minEig;
        double coeff = 2.0*q_rho*q_h*q_h*fabs(mu);
        if (mu < 0) {
          coeff += 0.5*q_rho*q_h*soundSpeed;
        }
        for (int y = 0; y < NUM_DIM; ++y) {
          for (int x = 0; x < NUM_DIM; ++x) {
            q_stress[ijN(x,y,2)] += coeff*q_gradv[ijN(x,y,2)];
          }
        }
      }
      const double S00 = q_stress[ijN(0,0,2)]; const double S10 = q_stress[ijN(1,0,2)];
      const double S01 = q_stress[ijN(0,1,2)]; const double S11 = q_stress[ijN(1,1,2)];
      stressJinvT[ijklNM(0,0,q,el,NUM_DIM,NUM_QUAD)] = q_Jw*((S00*invJ_00)+(S10*invJ_01));
      stressJinvT[ijklNM(1,0,q,el,NUM_DIM,NUM_QUAD)] = q_Jw*((S00*invJ_10)+(S10*invJ_11));
      stressJinvT[ijklNM(0,1,q,el,NUM_DIM,NUM_QUAD)] = q_Jw*((S01*invJ_00)+(S11*invJ_01));
      stressJinvT[ijklNM(1,1,q,el,NUM_DIM,NUM_QUAD)] = q_Jw*((S01*invJ_10)+(S11*invJ_11));
    }});
}

// *****************************************************************************
void rUpdateQuadratureData3D(const double GAMMA,
                             const double H0,
                             const double CFL,
                             const bool USE_VISCOSITY,
                             const int NUM_DIM,
                             const int NUM_QUAD,
                             const int NUM_QUAD_1D,
                             const int NUM_QUAD_2D,
                             const int NUM_QUAD_3D,
                             const int NUM_DOFS_1D,
                             const int numElements,
                             const double* dofToQuad,
                             const double* dofToQuadD,
                             const double* quadWeights,
                             const double* v,
                             const double* e,
                             const double* rho0DetJ0w,
                             const double* invJ0,
                             const double* J,
                             const double* invJ,
                             const double* detJ,
                             double* __restrict stressJinvT,
                             double* __restrict dtEst) {
  //assert(NUM_DIM==2); const int nd = 2;
  assert(NUM_QUAD_1D==2); const int q1 = 2;
  assert(NUM_QUAD_2D==4); const int q2 = 4;
  assert(NUM_QUAD_3D==8); const int q3 = 8;
  
  forall(el,numElements,{
    double s_gradv[9*q3];
    
    for (int i = 0; i < (9*NUM_QUAD_3D); ++i) {
      s_gradv[i] = 0;
    }

    for (int dz = 0; dz < NUM_DOFS_1D; ++dz) {
      double vDxy[3*q2] ;
      double vxDy[3*q2] ;
      double vxy[3*q2]  ;
      for (int i = 0; i < (3*NUM_QUAD_2D); ++i) {
        vDxy[i] = 0;
        vxDy[i] = 0;
        vxy[i]  = 0;
      }

      for (int dy = 0; dy < NUM_DOFS_1D; ++dy) {
        double vDx[3*q1] ;
        double vx[3*q1]  ;
        for (int i = 0; i < (3*NUM_QUAD_1D); ++i) {
          vDx[i] = 0;
          vx[i]  = 0;
        }

        for (int dx = 0; dx < NUM_DOFS_1D; ++dx) {
          for (int qx = 0; qx < NUM_QUAD_1D; ++qx) {
            for (int vi = 0; vi < 3; ++vi) {
              vDx[ijN(vi,qx,3)] += v[_ijklmNM(vi,dx,dy,dz,el,NUM_DOFS_1D,numElements)]*dofToQuadD[ijN(qx,dx,NUM_QUAD_1D)];
              vx[ijN(vi,qx,3)]  += v[_ijklmNM(vi,dx,dy,dz,el,NUM_DOFS_1D,numElements)]*dofToQuad[ijN(qx,dx,NUM_QUAD_1D)];
            }
          }
        }

        for (int qy = 0; qy < NUM_QUAD_1D; ++qy) {
          const double wy  = dofToQuad[ijN(qy,dy,NUM_QUAD_1D)];
          const double wDy = dofToQuadD[ijN(qy,dy,NUM_QUAD_1D)];
          for (int qx = 0; qx < NUM_QUAD_1D; ++qx) {
            for (int vi = 0; vi < 3; ++vi) {
              vDxy[ijkNM(vi,qx,qy,3,NUM_QUAD_1D)] += wy *vDx[ijN(vi,qx,3)];
              vxDy[ijkNM(vi,qx,qy,3,NUM_QUAD_1D)] += wDy*vx[ijN(vi,qx,3)];
              vxy[ijkNM(vi,qx,qy,3,NUM_QUAD_1D)]  += wy *vx[ijN(vi,qx,3)];
            }
          }
        }
      }
      for (int qz = 0; qz < NUM_DOFS_1D; ++qz) {
        const double wz  = dofToQuad[ijN(qz,dz,NUM_QUAD_1D)];
        const double wDz = dofToQuadD[ijN(qz,dz,NUM_QUAD_1D)];
        for (int qy = 0; qy < NUM_QUAD_1D; ++qy) {
          for (int qx = 0; qx < NUM_QUAD_1D; ++qx) {
            const int q = qx+qy*NUM_QUAD_1D+qz*NUM_QUAD_2D;
            for (int vi = 0; vi < 3; ++vi) {
              s_gradv[ijkNM(vi,0,q,3,NUM_DOFS_1D)] += wz *vDxy[ijkNM(vi,qx,qy,3,NUM_QUAD_1D)];
              s_gradv[ijkNM(vi,1,q,3,NUM_DOFS_1D)] += wz *vxDy[ijkNM(vi,qx,qy,3,NUM_QUAD_1D)];
              s_gradv[ijkNM(vi,2,q,3,NUM_DOFS_1D)] += wDz*vxy[ijkNM(vi,qx,qy,3,NUM_QUAD_1D)];
            }
          }
        }
      }
    }

    for (int q = 0; q < NUM_QUAD; ++q) {
      double q_gradv[9]  ;
      double q_stress[9] ;

      const double invJ_00 = invJ[ijklNM(0,0,q,el,NUM_DIM,NUM_QUAD)];
      const double invJ_10 = invJ[ijklNM(1,0,q,el,NUM_DIM,NUM_QUAD)];
      const double invJ_20 = invJ[ijklNM(2,0,q,el,NUM_DIM,NUM_QUAD)];
      const double invJ_01 = invJ[ijklNM(0,1,q,el,NUM_DIM,NUM_QUAD)];
      const double invJ_11 = invJ[ijklNM(1,1,q,el,NUM_DIM,NUM_QUAD)];
      const double invJ_21 = invJ[ijklNM(2,1,q,el,NUM_DIM,NUM_QUAD)];
      const double invJ_02 = invJ[ijklNM(0,2,q,el,NUM_DIM,NUM_QUAD)];
      const double invJ_12 = invJ[ijklNM(1,2,q,el,NUM_DIM,NUM_QUAD)];
      const double invJ_22 = invJ[ijklNM(2,2,q,el,NUM_DIM,NUM_QUAD)];

      q_gradv[ijN(0,0,3)] = ((s_gradv[ijkNM(0,0,q,3,NUM_DOFS_1D)]*invJ_00) +
                             (s_gradv[ijkNM(1,0,q,3,NUM_DOFS_1D)]*invJ_01) +
                             (s_gradv[ijkNM(2,0,q,3,NUM_DOFS_1D)]*invJ_02));
      q_gradv[ijN(1,0,3)] = ((s_gradv[ijkNM(0,0,q,3,NUM_DOFS_1D)]*invJ_10) +
                             (s_gradv[ijkNM(1,0,q,3,NUM_DOFS_1D)]*invJ_11) +
                             (s_gradv[ijkNM(2,0,q,3,NUM_DOFS_1D)]*invJ_12));
      q_gradv[ijN(2,0,3)] = ((s_gradv[ijkNM(0,0,q,3,NUM_DOFS_1D)]*invJ_20) +
                             (s_gradv[ijkNM(1,0,q,3,NUM_DOFS_1D)]*invJ_21) +
                             (s_gradv[ijkNM(2,0,q,3,NUM_DOFS_1D)]*invJ_22));

      q_gradv[ijN(0,1,3)] = ((s_gradv[ijkNM(0,1,q,3,NUM_DOFS_1D)]*invJ_00) +
                             (s_gradv[ijkNM(1,1,q,3,NUM_DOFS_1D)]*invJ_01) +
                             (s_gradv[ijkNM(2,1,q,3,NUM_DOFS_1D)]*invJ_02));
      q_gradv[ijN(1,1,3)] = ((s_gradv[ijkNM(0,1,q,3,NUM_DOFS_1D)]*invJ_10) +
                             (s_gradv[ijkNM(1,1,q,3,NUM_DOFS_1D)]*invJ_11) +
                             (s_gradv[ijkNM(2,1,q,3,NUM_DOFS_1D)]*invJ_12));
      q_gradv[ijN(2,1,3)] = ((s_gradv[ijkNM(0,1,q,3,NUM_DOFS_1D)]*invJ_20) +
                             (s_gradv[ijkNM(1,1,q,3,NUM_DOFS_1D)]*invJ_21) +
                             (s_gradv[ijkNM(2,1,q,3,NUM_DOFS_1D)]*invJ_22));

      q_gradv[ijN(0,2,3)] = ((s_gradv[ijkNM(0,2,q,3,NUM_DOFS_1D)]*invJ_00) +
                             (s_gradv[ijkNM(1,2,q,3,NUM_DOFS_1D)]*invJ_01) +
                             (s_gradv[ijkNM(2,2,q,3,NUM_DOFS_1D)]*invJ_02));
      q_gradv[ijN(1,2,3)] = ((s_gradv[ijkNM(0,2,q,3,NUM_DOFS_1D)]*invJ_10) +
                             (s_gradv[ijkNM(1,2,q,3,NUM_DOFS_1D)]*invJ_11) +
                             (s_gradv[ijkNM(2,2,q,3,NUM_DOFS_1D)]*invJ_12));
      q_gradv[ijN(2,2,3)] = ((s_gradv[ijkNM(0,2,q,3,NUM_DOFS_1D)]*invJ_20) +
                             (s_gradv[ijkNM(1,2,q,3,NUM_DOFS_1D)]*invJ_21) +
                             (s_gradv[ijkNM(2,2,q,3,NUM_DOFS_1D)]*invJ_22));

      const double q_Jw = detJ[ijN(q,el,NUM_QUAD)]*quadWeights[q];

      const double q_rho = rho0DetJ0w[ijN(q,el,NUM_QUAD)] / q_Jw;
      const double q_e   = fmax(0.0,e[ijN(q,el,NUM_QUAD)]);

      const double s = -(GAMMA-1.0)*q_rho*q_e;
      q_stress[ijN(0,0,3)] = s; q_stress[ijN(1,0,3)] = 0; q_stress[ijN(2,0,3)] = 0;
      q_stress[ijN(0,1,3)] = 0; q_stress[ijN(1,1,3)] = s; q_stress[ijN(2,1,3)] = 0;
      q_stress[ijN(0,2,3)] = 0; q_stress[ijN(1,2,3)] = 0; q_stress[ijN(2,2,3)] = s;

      const double gradv00 = q_gradv[ijN(0,0,3)];
      const double gradv11 = q_gradv[ijN(1,1,3)];
      const double gradv22 = q_gradv[ijN(2,2,3)];
      const double gradv10 = 0.5*(q_gradv[ijN(1,0,3)]+q_gradv[ijN(0,1,3)]);
      const double gradv20 = 0.5*(q_gradv[ijN(2,0,3)]+q_gradv[ijN(0,2,3)]);
      const double gradv21 = 0.5*(q_gradv[ijN(2,1,3)]+q_gradv[ijN(1,2,3)]);
      q_gradv[ijN(1,0,3)] = gradv10; q_gradv[ijN(2,0,3)] = gradv20;
      q_gradv[ijN(0,1,3)] = gradv10; q_gradv[ijN(2,1,3)] = gradv21;
      q_gradv[ijN(0,2,3)] = gradv20; q_gradv[ijN(1,2,3)] = gradv21;

      double minEig = 0;
      double comprDirX = 1;
      double comprDirY = 0;
      double comprDirZ = 0;

      {
        // Compute eigenvalues using quadrature formula
        const double q_ = (gradv00+gradv11+gradv22) / 3.0;
        const double gradv_q00 = (gradv00-q_);
        const double gradv_q11 = (gradv11-q_);
        const double gradv_q22 = (gradv22-q_);

        const double p1 = ((gradv10*gradv10) +
                           (gradv20*gradv20) +
                           (gradv21*gradv21));
        const double p2 = ((gradv_q00*gradv_q00) +
                           (gradv_q11*gradv_q11) +
                           (gradv_q22*gradv_q22) +
                           (2.0*p1));
        const double p    = sqrt(p2 / 6.0);
        const double pinv = 1.0 / p;
        // det(pinv*(gradv-q*I))
        const double r = (0.5*pinv*pinv*pinv *
                          ((gradv_q00*gradv_q11*gradv_q22) +
                           (2.0*gradv10*gradv21*gradv20) -
                           (gradv_q11*gradv20*gradv20) -
                           (gradv_q22*gradv10*gradv10) -
                           (gradv_q00*gradv21*gradv21)));

        double phi = 0;
        if (r <= -1.0) {
          phi = M_PI / 3.0;
        } else if (r < 1.0) {
          phi = acos(r) / 3.0;
        }

        minEig = q_+(2.0*p*cos(phi+(2.0*M_PI / 3.0)));
        const double eig3 = q_+(2.0*p*cos(phi));
        const double eig2 = 3.0*q_-minEig-eig3;
        double maxNorm = 0;

        for (int i = 0; i < 3; ++i) {
          const double x = q_gradv[i+3*0]-(i == 0)*eig3;
          const double y = q_gradv[i+3*1]-(i == 1)*eig3;
          const double z = q_gradv[i+3*2]-(i == 2)*eig3;
          const double cx = ((x*(gradv00-eig2)) +
                             (y*gradv10) +
                             (z*gradv20));
          const double cy = ((x*gradv10) +
                             (y*(gradv11-eig2)) +
                             (z*gradv21));
          const double cz = ((x*gradv20) +
                             (y*gradv21) +
                             (z*(gradv22-eig2)));
          const double cNorm = (cx*cx+cy*cy+cz*cz);
          //#warning 1e-16 to 1
          if ((cNorm > 1.e-16) && (maxNorm < cNorm)) {
            comprDirX = cx;
            comprDirY = cy;
            comprDirZ = cz;
            maxNorm = cNorm;
          }
        }
        //#warning 1e-16 to 1
        if (maxNorm > 1.e-16) {
          const double maxNormInv = 1.0 / sqrt(maxNorm);
          comprDirX *= maxNormInv;
          comprDirY *= maxNormInv;
          comprDirZ *= maxNormInv;
        }
      }

      // Computes the initial->physical transformation Jacobian.
      const double J_00 = J[ijklNM(0,0,q,el,NUM_DIM,NUM_QUAD)];
      const double J_10 = J[ijklNM(1,0,q,el,NUM_DIM,NUM_QUAD)];
      const double J_20 = J[ijklNM(2,0,q,el,NUM_DIM,NUM_QUAD)];
      const double J_01 = J[ijklNM(0,1,q,el,NUM_DIM,NUM_QUAD)];
      const double J_11 = J[ijklNM(1,1,q,el,NUM_DIM,NUM_QUAD)];
      const double J_21 = J[ijklNM(2,1,q,el,NUM_DIM,NUM_QUAD)];
      const double J_02 = J[ijklNM(0,2,q,el,NUM_DIM,NUM_QUAD)];
      const double J_12 = J[ijklNM(1,2,q,el,NUM_DIM,NUM_QUAD)];
      const double J_22 = J[ijklNM(2,2,q,el,NUM_DIM,NUM_QUAD)];

      const double invJ0_00 = invJ0[ijklNM(0,0,q,el,NUM_DIM,NUM_QUAD)];
      const double invJ0_10 = invJ0[ijklNM(1,0,q,el,NUM_DIM,NUM_QUAD)];
      const double invJ0_20 = invJ0[ijklNM(2,0,q,el,NUM_DIM,NUM_QUAD)];
      const double invJ0_01 = invJ0[ijklNM(0,1,q,el,NUM_DIM,NUM_QUAD)];
      const double invJ0_11 = invJ0[ijklNM(1,1,q,el,NUM_DIM,NUM_QUAD)];
      const double invJ0_21 = invJ0[ijklNM(2,1,q,el,NUM_DIM,NUM_QUAD)];
      const double invJ0_02 = invJ0[ijklNM(0,2,q,el,NUM_DIM,NUM_QUAD)];
      const double invJ0_12 = invJ0[ijklNM(1,2,q,el,NUM_DIM,NUM_QUAD)];
      const double invJ0_22 = invJ0[ijklNM(2,2,q,el,NUM_DIM,NUM_QUAD)];

      const double Jpi_00 = ((J_00*invJ0_00)+(J_10*invJ0_01)+(J_20*invJ0_02));
      const double Jpi_10 = ((J_00*invJ0_10)+(J_10*invJ0_11)+(J_20*invJ0_12));
      const double Jpi_20 = ((J_00*invJ0_20)+(J_10*invJ0_21)+(J_20*invJ0_22));

      const double Jpi_01 = ((J_01*invJ0_00)+(J_11*invJ0_01)+(J_21*invJ0_02));
      const double Jpi_11 = ((J_01*invJ0_10)+(J_11*invJ0_11)+(J_21*invJ0_12));
      const double Jpi_21 = ((J_01*invJ0_20)+(J_11*invJ0_21)+(J_21*invJ0_22));

      const double Jpi_02 = ((J_02*invJ0_00)+(J_12*invJ0_01)+(J_22*invJ0_02));
      const double Jpi_12 = ((J_02*invJ0_10)+(J_12*invJ0_11)+(J_22*invJ0_12));
      const double Jpi_22 = ((J_02*invJ0_20)+(J_12*invJ0_21)+(J_22*invJ0_22));

      const double physDirX = ((Jpi_00*comprDirX)+(Jpi_10*comprDirY)+(Jpi_20*comprDirZ));
      const double physDirY = ((Jpi_01*comprDirX)+(Jpi_11*comprDirY)+(Jpi_21*comprDirZ));
      const double physDirZ = ((Jpi_02*comprDirX)+(Jpi_12*comprDirY)+(Jpi_22*comprDirZ));

      const double q_h = H0*sqrt((physDirX*physDirX)+
                                 (physDirY*physDirY)+
                                 (physDirZ*physDirZ));

      const double soundSpeed = sqrt(GAMMA*(GAMMA-1.0)*q_e);
      dtEst[ijN(q,el,NUM_QUAD)] = CFL*q_h / soundSpeed;

      if (USE_VISCOSITY) {
        // TODO: Check how we can extract outside of kernel
        const double mu = minEig;
        double coeff = 2.0*q_rho*q_h*q_h*fabs(mu);
        if (mu < 0) {
          coeff += 0.5*q_rho*q_h*soundSpeed;
        }
        for (int y = 0; y < 3; ++y) {
          for (int x = 0; x < 3; ++x) {
            q_stress[ijN(x,y,3)] += coeff*q_gradv[ijN(x,y,3)];
          }
        }
      }

      const double S00 = q_stress[ijN(0,0,3)];
      const double S10 = q_stress[ijN(1,0,3)];
      const double S20 = q_stress[ijN(2,0,3)];
      const double S01 = q_stress[ijN(0,1,3)];
      const double S11 = q_stress[ijN(1,1,3)];
      const double S21 = q_stress[ijN(2,1,3)];
      const double S02 = q_stress[ijN(0,2,3)];
      const double S12 = q_stress[ijN(1,2,3)];
      const double S22 = q_stress[ijN(2,2,3)];

      stressJinvT[ijklNM(0,0,q,el,NUM_DIM,NUM_QUAD)] = q_Jw*((S00*invJ_00)+(S10*invJ_01)+(S20*invJ_02));
      stressJinvT[ijklNM(1,0,q,el,NUM_DIM,NUM_QUAD)] = q_Jw*((S00*invJ_10)+(S10*invJ_11)+(S20*invJ_12));
      stressJinvT[ijklNM(2,0,q,el,NUM_DIM,NUM_QUAD)] = q_Jw*((S00*invJ_20)+(S10*invJ_21)+(S20*invJ_22));

      stressJinvT[ijklNM(0,1,q,el,NUM_DIM,NUM_QUAD)] = q_Jw*((S01*invJ_00)+(S11*invJ_01)+(S21*invJ_02));
      stressJinvT[ijklNM(1,1,q,el,NUM_DIM,NUM_QUAD)] = q_Jw*((S01*invJ_10)+(S11*invJ_11)+(S21*invJ_12));
      stressJinvT[ijklNM(2,1,q,el,NUM_DIM,NUM_QUAD)] = q_Jw*((S01*invJ_20)+(S11*invJ_21)+(S21*invJ_22));

      stressJinvT[ijklNM(0,2,q,el,NUM_DIM,NUM_QUAD)] = q_Jw*((S02*invJ_00)+(S12*invJ_01)+(S22*invJ_02));
      stressJinvT[ijklNM(1,2,q,el,NUM_DIM,NUM_QUAD)] = q_Jw*((S02*invJ_10)+(S12*invJ_11)+(S22*invJ_12));
      stressJinvT[ijklNM(2,2,q,el,NUM_DIM,NUM_QUAD)] = q_Jw*((S02*invJ_20)+(S12*invJ_21)+(S22*invJ_22));
    }
    });
}

// *****************************************************************************
void rUpdateQuadratureData(const double GAMMA,
                           const double H0,
                           const double CFL,
                           const bool USE_VISCOSITY,
                           const int NUM_DIM,
                           const int NUM_QUAD,
                           const int NUM_QUAD_1D,
                           const int NUM_DOFS_1D,
                           const int nzones,
                           const double* dofToQuad,
                           const double* dofToQuadD,
                           const double* quadWeights,
                           const double* v,
                           const double* e,
                           const double* rho0DetJ0w,
                           const double* invJ0,
                           const double* J,
                           const double* invJ,
                           const double* detJ,
                           double* __restrict stressJinvT,
                           double* __restrict dtEst){
   if (NUM_DIM==1) { assert(false); }
   if (NUM_DIM==2)
     rUpdateQuadratureData2D(GAMMA,H0,CFL,USE_VISCOSITY,NUM_DIM,
                             NUM_QUAD,
                             NUM_QUAD_1D,
                             NUM_QUAD_1D*NUM_QUAD_1D,
                             NUM_DOFS_1D,
                             nzones,
                             dofToQuad,
                             dofToQuadD,
                             quadWeights,
                             v,
                             e,
                             rho0DetJ0w,
                             invJ0,
                             J,
                             invJ,
                             detJ,
                             stressJinvT,
                             dtEst);
   if (NUM_DIM==3)
     rUpdateQuadratureData3D(GAMMA,H0,CFL,USE_VISCOSITY,NUM_DIM,
                             NUM_QUAD,
                             NUM_QUAD_1D,
                             NUM_QUAD_1D*NUM_QUAD_1D,
                             NUM_QUAD_1D*NUM_QUAD_1D*NUM_QUAD_1D,
                             NUM_DOFS_1D,
                             nzones,
                             dofToQuad,
                             dofToQuadD,
                             quadWeights,
                             v,
                             e,
                             rho0DetJ0w,
                             invJ0,
                             J,
                             invJ,
                             detJ,
                             stressJinvT,
                             dtEst);
}
