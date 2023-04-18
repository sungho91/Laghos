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
//
//                     __                __
//                    / /   ____  ____  / /_  ____  _____
//                   / /   / __ `/ __ `/ __ \/ __ \/ ___/
//                  / /___/ /_/ / /_/ / / / / /_/ (__  )
//                 /_____/\__,_/\__, /_/ /_/\____/____/
//                             /____/
//
//             High-order Lagrangian Geodynamics Solver
//
// Laghos(LAGrangian High-Order Solver) is a miniapp that solves the
// time-dependent Euler equation of compressible gas dynamics in a moving
// Lagrangian frame using unstructured high-order finite element spatial
// discretization and explicit high-order time-stepping. Laghos is based on the
// numerical algorithm described in the following article:
//
//    V. Dobrev, Tz. Kolev and R. Rieben, "High-order curvilinear finite element
//    methods for Lagrangian geodynamics", SIAM Journal on Scientific
//    Computing, (34) 2012, pp. B606–B641, https://doi.org/10.1137/120864672.
//
// Test problems:
//    p = 0  --> Taylor-Green vortex (smooth problem).
//    p = 1  --> Sedov blast.
//    p = 2  --> 1D Sod shock tube.
//    p = 3  --> Triple point.
//    p = 4  --> Gresho vortex (smooth problem).
//    p = 5  --> 2D Riemann problem, config. 12 of doi.org/10.1002/num.10025
//    p = 6  --> 2D Riemann problem, config.  6 of doi.org/10.1002/num.10025
//    p = 7  --> 2D Rayleigh-Taylor instability problem.
//
// Sample runs: see README.md, section 'Verification of Results'.
//
// Combinations resulting in 3D uniform Cartesian MPI partitionings of the mesh:
// -m data/cube01_hex.mesh   -pt 211 for  2 / 16 / 128 / 1024 ... tasks.
// -m data/cube_922_hex.mesh -pt 921 for    / 18 / 144 / 1152 ... tasks.
// -m data/cube_522_hex.mesh -pt 522 for    / 20 / 160 / 1280 ... tasks.
// -m data/cube_12_hex.mesh  -pt 311 for  3 / 24 / 192 / 1536 ... tasks.
// -m data/cube01_hex.mesh   -pt 221 for  4 / 32 / 256 / 2048 ... tasks.
// -m data/cube_922_hex.mesh -pt 922 for    / 36 / 288 / 2304 ... tasks.
// -m data/cube_522_hex.mesh -pt 511 for  5 / 40 / 320 / 2560 ... tasks.
// -m data/cube_12_hex.mesh  -pt 321 for  6 / 48 / 384 / 3072 ... tasks.
// -m data/cube01_hex.mesh   -pt 111 for  8 / 64 / 512 / 4096 ... tasks.
// -m data/cube_922_hex.mesh -pt 911 for  9 / 72 / 576 / 4608 ... tasks.
// -m data/cube_522_hex.mesh -pt 521 for 10 / 80 / 640 / 5120 ... tasks.
// -m data/cube_12_hex.mesh  -pt 322 for 12 / 96 / 768 / 6144 ... tasks.
// mpirun -np 8 laghos -p 1 -fa -dim 2 -rs 2 -tf 50e3 -m ../mesh_data/Qmesh2d.mesh 2D beam (80 * 10 km2)
// mpirun -np 8 laghos -p 1 -fa -dim 2 -rs 2 -tf 50e3 -m ../mesh_data/Qmesh3d.mesh 3D beam (80 * 10 * 10 km3)

#include <fstream>
#include <sys/time.h>
#include <sys/resource.h>
#include "laghos_solver.hpp"
#include <cmath>

using std::cout;
using std::endl;
using namespace mfem;

// Choice for the problem setup.
static int problem, dim;

// Forward declarations.
double e0(const Vector &);
double p0(const Vector &);
double rho0(const Vector &);
double gamma_func(const Vector &);
void v0(const Vector &, Vector &);

void Returnmapping (Vector &, Vector &, Vector &, int &, Vector &, Vector &, Vector &, Vector &, Vector &, Vector &);

static long GetMaxRssMB();
static void display_banner(std::ostream&);
static void Checks(const int ti, const double norm, int &checks);

class StressCoefficient : public VectorCoefficient
{
private:
   ParGridFunction &u;
   Coefficient &lambda, &mu;
   DenseMatrix eps, sigma;
   int dim;

public:
   StressCoefficient (int &_dim, ParGridFunction &_u, Coefficient &_lambda, Coefficient &_mu)
      : VectorCoefficient(_dim), u(_u), lambda(_lambda), mu(_mu) {dim=_dim;}
   virtual void Eval(Vector &K, ElementTransformation &T, const IntegrationPoint &ip)
   {
   u.GetVectorGradient(T, eps);  // eps = grad(u)
   eps.Symmetrize();             // eps = (1/2)*(grad(u) + grad(u)^t)
   double l = lambda.Eval(T, ip);
   double m = mu.Eval(T, ip);
   sigma=0.0;
   sigma.Diag(2*eps.Trace()/3, eps.Size()); // sigma = lambda*trace(eps)*I
   sigma.Add(2, eps);          // sigma += 2*mu*eps
   
   K.SetSize(3*(dim-1));
   if(dim ==2)
   {
      K(0)=sigma(0,0); K(1)=sigma(1,1); K(2)=sigma(0,1);
   }
   else if(dim ==3)
   {
      K(0)=0; K(1)=1; K(2)=2;
      K(3)=3; K(4)=4; K(5)=5;

      // K(0)=sigma(0,0); K(1)=sigma(1,1); K(2)=sigma(2,2);
      // K(3)=sigma(0,1); K(4)=sigma(0,2); K(5)=sigma(1,2);
   }
   }

   virtual ~StressCoefficient() { }
};

int main(int argc, char *argv[])
{
   // Initialize MPI.
   mfem::MPI_Session mpi(argc, argv);
   const int myid = mpi.WorldRank();

   // Print the banner.
   if (mpi.Root()) { display_banner(cout); }

   // Parse command-line options.
   problem = 1;
   dim = 3;
   const char *mesh_file = "default";
   int rs_levels = 2;
   int rp_levels = 0;
   Array<int> cxyz;
   int order_v = 2;
   int order_e = 1;
   // int order_q = 3;
   int order_q = -1;
   int ode_solver_type = 7;
   double t_final = 1.0;
   double cfl = 0.5;
   double cg_tol = 1e-12;
   double ftz_tol = 0.0;
   int cg_max_iter = 300;
   int max_tsteps = -1;
   bool p_assembly = false;
   bool impose_visc = true;
   bool visualization = false;
   int vis_steps = 1000;
   bool visit = false;
   bool paraview = true;
   bool gfprint = false;
   const char *basename = "results/Laghos";
   int partition_type = 0;
   const char *device = "cpu";
   bool check = false;
   bool mem_usage = false;
   bool fom = false;
   bool gpu_aware_mpi = false;
   int dev = 0;
   bool year = false;
   // bool year = true;
   double init_dt = 1.0;
   // double mscale  = 1.0;
   double mscale  = 1.0e6;
   double gravity = 0.0; // magnitude 
   // double init_dt = 1e-1;
   double blast_energy = 0.0;
   // double blast_energy = 1.0e-6;
   double blast_position[] = {0.0, 0.5, 0.0};
   // double blast_energy2 = 0.1;
   // double blast_position2[] = {8.0, 0.5};

   OptionsParser args(argc, argv);
   args.AddOption(&dim, "-dim", "--dimension", "Dimension of the problem.");
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
   args.AddOption(&rs_levels, "-rs", "--refine-serial",
                  "Number of times to refine the mesh uniformly in serial.");
   args.AddOption(&rp_levels, "-rp", "--refine-parallel",
                  "Number of times to refine the mesh uniformly in parallel.");
   args.AddOption(&cxyz, "-c", "--cartesian-partitioning",
                  "Use Cartesian partitioning.");
   args.AddOption(&problem, "-p", "--problem", "Problem setup to use.");
   args.AddOption(&order_v, "-ok", "--order-kinematic",
                  "Order (degree) of the kinematic finite element space.");
   args.AddOption(&order_e, "-ot", "--order-thermo",
                  "Order (degree) of the thermodynamic finite element space.");
   args.AddOption(&order_q, "-oq", "--order-intrule",
                  "Order  of the integration rule.");
   args.AddOption(&ode_solver_type, "-s", "--ode-solver",
                  "ODE solver: 1 - Forward Euler,\n\t"
                  "            2 - RK2 SSP, 3 - RK3 SSP, 4 - RK4, 6 - RK6,\n\t"
                  "            7 - RK2Avg.");
   args.AddOption(&t_final, "-tf", "--t-final",
                  "Final time; start time is 0.");
   args.AddOption(&cfl, "-cfl", "--cfl", "CFL-condition number.");
   args.AddOption(&cg_tol, "-cgt", "--cg-tol",
                  "Relative CG tolerance (velocity linear solve).");
   args.AddOption(&ftz_tol, "-ftz", "--ftz-tol",
                  "Absolute flush-to-zero tolerance.");
   args.AddOption(&cg_max_iter, "-cgm", "--cg-max-steps",
                  "Maximum number of CG iterations (velocity linear solve).");
   args.AddOption(&max_tsteps, "-ms", "--max-steps",
                  "Maximum number of steps (negative means no restriction).");
   args.AddOption(&p_assembly, "-pa", "--partial-assembly", "-fa",
                  "--full-assembly",
                  "Activate 1D tensor-based assembly (partial assembly).");
   args.AddOption(&impose_visc, "-iv", "--impose-viscosity", "-niv",
                  "--no-impose-viscosity",
                  "Use active viscosity terms even for smooth problems.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&vis_steps, "-vs", "--visualization-steps",
                  "Visualize every n-th timestep.");
   args.AddOption(&visit, "-visit", "--visit", "-no-visit", "--no-visit",
                  "Enable or disable VisIt visualization.");
   args.AddOption(&paraview, "-paraview", "--paraview-datafiles", "-no-paraview",
                  "--no-paraview-datafiles",
                  "Save data files for ParaView (paraview.org) visualization.");
   args.AddOption(&gfprint, "-print", "--print", "-no-print", "--no-print",
                  "Enable or disable result output (files in mfem format).");
   args.AddOption(&basename, "-k", "--outputfilename",
                  "Name of the visit dump files");
   args.AddOption(&partition_type, "-pt", "--partition",
                  "Customized x/y/z Cartesian MPI partitioning of the serial mesh.\n\t"
                  "Here x,y,z are relative task ratios in each direction.\n\t"
                  "Example: with 48 mpi tasks and -pt 321, one would get a Cartesian\n\t"
                  "partition of the serial mesh by (6,4,2) MPI tasks in (x,y,z).\n\t"
                  "NOTE: the serially refined mesh must have the appropriate number\n\t"
                  "of zones in each direction, e.g., the number of zones in direction x\n\t"
                  "must be divisible by the number of MPI tasks in direction x.\n\t"
                  "Available options: 11, 21, 111, 211, 221, 311, 321, 322, 432.");
   args.AddOption(&device, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.AddOption(&check, "-chk", "--checks", "-no-chk", "--no-checks",
                  "Enable 2D checks.");
   args.AddOption(&mem_usage, "-mb", "--mem", "-no-mem", "--no-mem",
                  "Enable memory usage.");
   args.AddOption(&fom, "-f", "--fom", "-no-fom", "--no-fom",
                  "Enable figure of merit output.");
   args.AddOption(&gpu_aware_mpi, "-gam", "--gpu-aware-mpi", "-no-gam",
                  "--no-gpu-aware-mpi", "Enable GPU aware MPI communications.");
   args.AddOption(&dev, "-dev", "--dev", "GPU device to use.");
   args.Parse();
   if (!args.Good())
   {
      if (mpi.Root()) { args.PrintUsage(cout); }
      return 1;
   }
   if (mpi.Root()) { args.PrintOptions(cout); }
   
   if(max_tsteps > -1)
   {
      t_final = 1.0e38;
   }   
   if(year)
   {
      t_final = t_final * 86400 * 365.25;
      
      if (mpi.Root())
      {
         std::cout << "Use years in output instead of seconds is true" << std::endl;
      }
   }
   else
   {
      // init_dt = 1e-1;
      if (mpi.Root())
      {
         std::cout << "Use seconds in output instead of years is true" << std::endl;
      }
      
   }

   // Configure the device from the command line options
   Device backend;
   backend.Configure(device, dev);
   if (mpi.Root()) { backend.Print(); }
   backend.SetGPUAwareMPI(gpu_aware_mpi);

   // On all processors, use the default builtin 1D/2D/3D mesh or read the
   // serial one given on the command line.
   Mesh *mesh;

   if (strncmp(mesh_file, "default", 7) != 0)
   {
      mesh = new Mesh(mesh_file, true, true);
   }
   else
   {
      if (dim == 1)
      {
         mesh = new Mesh(Mesh::MakeCartesian1D(2));
         mesh->GetBdrElement(0)->SetAttribute(1);
         mesh->GetBdrElement(1)->SetAttribute(1);
      }
      if (dim == 2)
      {
         mesh = new Mesh(Mesh::MakeCartesian2D(2, 2, Element::QUADRILATERAL,
                                               true));
         const int NBE = mesh->GetNBE();
         for (int b = 0; b < NBE; b++)
         {
            Element *bel = mesh->GetBdrElement(b);
            const int attr = (b < NBE/2) ? 2 : 1;
            std::cout<< NBE <<"," << b << "," << attr <<std::endl;
            bel->SetAttribute(attr);
         }
      }
      if (dim == 3)
      {
         mesh = new Mesh(Mesh::MakeCartesian3D(2, 2, 2, Element::HEXAHEDRON,
                                               true));
         const int NBE = mesh->GetNBE();
         for (int b = 0; b < NBE; b++)
         {
            Element *bel = mesh->GetBdrElement(b);
            const int attr = (b < NBE/3) ? 3 : (b < 2*NBE/3) ? 1 : 2;
            bel->SetAttribute(attr);
         }
      }
   }
   dim = mesh->Dimension();

   // 1D vs partial assembly sanity check.
   if (p_assembly && dim == 1)
   {
      p_assembly = false;
      if (mpi.Root())
      {
         cout << "Laghos does not support PA in 1D. Switching to FA." << endl;
      }
   }

   // Refine the mesh in serial to increase the resolution.
   for (int lev = 0; lev < rs_levels; lev++) { mesh->UniformRefinement(); }
   const int mesh_NE = mesh->GetNE();
   if (mpi.Root())
   {
      cout << "Number of zones in the serial mesh: " << mesh_NE << endl;
   }

   // Parallel partitioning of the mesh.
   ParMesh *pmesh = nullptr;
   const int num_tasks = mpi.WorldSize(); int unit = 1;
   int *nxyz = new int[dim];
   switch (partition_type)
   {
      case 0:
         for (int d = 0; d < dim; d++) { nxyz[d] = unit; }
         break;
      case 11:
      case 111:
         unit = static_cast<int>(floor(pow(num_tasks, 1.0 / dim) + 1e-2));
         for (int d = 0; d < dim; d++) { nxyz[d] = unit; }
         break;
      case 21: // 2D
         unit = static_cast<int>(floor(pow(num_tasks / 2, 1.0 / 2) + 1e-2));
         nxyz[0] = 2 * unit; nxyz[1] = unit;
         break;
      case 31: // 2D
         unit = static_cast<int>(floor(pow(num_tasks / 3, 1.0 / 2) + 1e-2));
         nxyz[0] = 3 * unit; nxyz[1] = unit;
         break;
      case 32: // 2D
         unit = static_cast<int>(floor(pow(2 * num_tasks / 3, 1.0 / 2) + 1e-2));
         nxyz[0] = 3 * unit / 2; nxyz[1] = unit;
         break;
      case 49: // 2D
         unit = static_cast<int>(floor(pow(9 * num_tasks / 4, 1.0 / 2) + 1e-2));
         nxyz[0] = 4 * unit / 9; nxyz[1] = unit;
         break;
      case 51: // 2D
         unit = static_cast<int>(floor(pow(num_tasks / 5, 1.0 / 2) + 1e-2));
         nxyz[0] = 5 * unit; nxyz[1] = unit;
         break;
      case 211: // 3D.
         unit = static_cast<int>(floor(pow(num_tasks / 2, 1.0 / 3) + 1e-2));
         nxyz[0] = 2 * unit; nxyz[1] = unit; nxyz[2] = unit;
         break;
      case 221: // 3D.
         unit = static_cast<int>(floor(pow(num_tasks / 4, 1.0 / 3) + 1e-2));
         nxyz[0] = 2 * unit; nxyz[1] = 2 * unit; nxyz[2] = unit;
         break;
      case 311: // 3D.
         unit = static_cast<int>(floor(pow(num_tasks / 3, 1.0 / 3) + 1e-2));
         nxyz[0] = 3 * unit; nxyz[1] = unit; nxyz[2] = unit;
         break;
      case 321: // 3D.
         unit = static_cast<int>(floor(pow(num_tasks / 6, 1.0 / 3) + 1e-2));
         nxyz[0] = 3 * unit; nxyz[1] = 2 * unit; nxyz[2] = unit;
         break;
      case 322: // 3D.
         unit = static_cast<int>(floor(pow(2 * num_tasks / 3, 1.0 / 3) + 1e-2));
         nxyz[0] = 3 * unit / 2; nxyz[1] = unit; nxyz[2] = unit;
         break;
      case 432: // 3D.
         unit = static_cast<int>(floor(pow(num_tasks / 3, 1.0 / 3) + 1e-2));
         nxyz[0] = 2 * unit; nxyz[1] = 3 * unit / 2; nxyz[2] = unit;
         break;
      case 511: // 3D.
         unit = static_cast<int>(floor(pow(num_tasks / 5, 1.0 / 3) + 1e-2));
         nxyz[0] = 5 * unit; nxyz[1] = unit; nxyz[2] = unit;
         break;
      case 521: // 3D.
         unit = static_cast<int>(floor(pow(num_tasks / 10, 1.0 / 3) + 1e-2));
         nxyz[0] = 5 * unit; nxyz[1] = 2 * unit; nxyz[2] = unit;
         break;
      case 522: // 3D.
         unit = static_cast<int>(floor(pow(num_tasks / 20, 1.0 / 3) + 1e-2));
         nxyz[0] = 5 * unit; nxyz[1] = 2 * unit; nxyz[2] = 2 * unit;
         break;
      case 911: // 3D.
         unit = static_cast<int>(floor(pow(num_tasks / 9, 1.0 / 3) + 1e-2));
         nxyz[0] = 9 * unit; nxyz[1] = unit; nxyz[2] = unit;
         break;
      case 921: // 3D.
         unit = static_cast<int>(floor(pow(num_tasks / 18, 1.0 / 3) + 1e-2));
         nxyz[0] = 9 * unit; nxyz[1] = 2 * unit; nxyz[2] = unit;
         break;
      case 922: // 3D.
         unit = static_cast<int>(floor(pow(num_tasks / 36, 1.0 / 3) + 1e-2));
         nxyz[0] = 9 * unit; nxyz[1] = 2 * unit; nxyz[2] = 2 * unit;
         break;
      default:
         if (myid == 0)
         {
            cout << "Unknown partition type: " << partition_type << '\n';
         }
         delete mesh;
         MPI_Finalize();
         return 3;
   }
   int product = 1;
   for (int d = 0; d < dim; d++) { product *= nxyz[d]; }
   const bool cartesian_partitioning = (cxyz.Size()>0)?true:false;
   if (product == num_tasks || cartesian_partitioning)
   {
      if (cartesian_partitioning)
      {
         int cproduct = 1;
         for (int d = 0; d < dim; d++) { cproduct *= cxyz[d]; }
         MFEM_VERIFY(!cartesian_partitioning || cxyz.Size() == dim,
                     "Expected " << mesh->SpaceDimension() << " integers with the "
                     "option --cartesian-partitioning.");
         MFEM_VERIFY(!cartesian_partitioning || num_tasks == cproduct,
                     "Expected cartesian partitioning product to match number of ranks.");
      }
      int *partitioning = cartesian_partitioning ?
                          mesh->CartesianPartitioning(cxyz):
                          mesh->CartesianPartitioning(nxyz);
      pmesh = new ParMesh(MPI_COMM_WORLD, *mesh, partitioning);
      delete [] partitioning;
   }
   else
   {
      if (myid == 0)
      {
         cout << "Non-Cartesian partitioning through METIS will be used.\n";
#ifndef MFEM_USE_METIS
         cout << "MFEM was built without METIS. "
              << "Adjust the number of tasks to use a Cartesian split." << endl;
#endif
      }
#ifndef MFEM_USE_METIS
      return 1;
#endif
      pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   }
   delete [] nxyz;
   delete mesh;

   // Refine the mesh further in parallel to increase the resolution.
   for (int lev = 0; lev < rp_levels; lev++) { pmesh->UniformRefinement(); }

   int NE = pmesh->GetNE(), ne_min, ne_max;
   MPI_Reduce(&NE, &ne_min, 1, MPI_INT, MPI_MIN, 0, pmesh->GetComm());
   MPI_Reduce(&NE, &ne_max, 1, MPI_INT, MPI_MAX, 0, pmesh->GetComm());
   if (myid == 0)
   { cout << "Zones min/max: " << ne_min << " " << ne_max << endl; }

   // Define the parallel finite element spaces. We use:
   // - H1 (Gauss-Lobatto, continuous) for position and velocity.
   // - L2 (Bernstein, discontinuous) for specific internal energy.
   L2_FECollection L2FEC(order_e, dim, BasisType::Positive);
   H1_FECollection H1FEC(order_v, dim);
   ParFiniteElementSpace L2FESpace(pmesh, &L2FEC);
   ParFiniteElementSpace H1FESpace(pmesh, &H1FEC, pmesh->Dimension());
   ParFiniteElementSpace L2FESpace_stress(pmesh, &L2FEC, 3*(dim-1)); // three varibles for 2D, six varibles for 3D
   
   // Boundary conditions: all tests use v.n = 0 on the boundary, and we assume
   // that the boundaries are straight.
   Array<int> ess_tdofs, ess_vdofs;
   {
      Array<int> ess_bdr(pmesh->bdr_attributes.Max()), dofs_marker, dofs_list;
      /*
      ess_bdr = 0; ess_bdr[0] = 1; 
      H1FESpace.GetEssentialTrueDofs(ess_bdr, dofs_list);
      ess_tdofs.Append(dofs_list);
      H1FESpace.GetEssentialVDofs(ess_bdr, dofs_marker);
      FiniteElementSpace::MarkerToList(dofs_marker, dofs_list);
      ess_vdofs.Append(dofs_list);

      ess_bdr = 0; ess_bdr[2] = 1; ess_bdr[2] = 1;
      H1FESpace.GetEssentialTrueDofs(ess_bdr, dofs_list, 0);
      ess_tdofs.Append(dofs_list);
      H1FESpace.GetEssentialVDofs(ess_bdr, dofs_marker, 0);
      FiniteElementSpace::MarkerToList(dofs_marker, dofs_list);
      ess_vdofs.Append(dofs_list);
      */
      
      /*
      // Body force test problem
      // x compoent is constained in left and right sides 
      ess_bdr = 0; ess_bdr[0] = 1; ess_bdr[1] = 1;
      H1FESpace.GetEssentialTrueDofs(ess_bdr, dofs_list, 0);
      ess_tdofs.Append(dofs_list);
      H1FESpace.GetEssentialVDofs(ess_bdr, dofs_marker, 0);
      FiniteElementSpace::MarkerToList(dofs_marker, dofs_list);
      ess_vdofs.Append(dofs_list);

      // Bottom is fixed
      ess_bdr = 0; ess_bdr[2] = 1; 
      H1FESpace.GetEssentialTrueDofs(ess_bdr, dofs_list);
      ess_tdofs.Append(dofs_list);
      H1FESpace.GetEssentialVDofs(ess_bdr, dofs_marker);
      FiniteElementSpace::MarkerToList(dofs_marker, dofs_list);
      ess_vdofs.Append(dofs_list);

      // y compoent is constained in left and right sides 
      ess_bdr = 0; ess_bdr[4] = 1; ess_bdr[5] = 1;
      H1FESpace.GetEssentialTrueDofs(ess_bdr, dofs_list, 1);
      ess_tdofs.Append(dofs_list);
      H1FESpace.GetEssentialVDofs(ess_bdr, dofs_marker, 1);
      FiniteElementSpace::MarkerToList(dofs_marker, dofs_list);
      ess_vdofs.Append(dofs_list);
      */

   
      // x compoent is constained in left and right sides 
      ess_bdr = 0; ess_bdr[0] = 1; ess_bdr[1] = 1;
      H1FESpace.GetEssentialTrueDofs(ess_bdr, dofs_list, 0);
      ess_tdofs.Append(dofs_list);
      H1FESpace.GetEssentialVDofs(ess_bdr, dofs_marker, 0);
      FiniteElementSpace::MarkerToList(dofs_marker, dofs_list);
      ess_vdofs.Append(dofs_list);

      // z compoent is constained in top and bottom sides
      ess_bdr = 0; ess_bdr[2] = 1; ess_bdr[3] = 1;
      H1FESpace.GetEssentialTrueDofs(ess_bdr, dofs_list, 2);
      ess_tdofs.Append(dofs_list);
      H1FESpace.GetEssentialVDofs(ess_bdr, dofs_marker, 2);
      FiniteElementSpace::MarkerToList(dofs_marker, dofs_list);
      ess_vdofs.Append(dofs_list);

      // y compoent is constained in front and back sides 
      ess_bdr = 0; ess_bdr[4] = 1; ess_bdr[5] = 1;
      H1FESpace.GetEssentialTrueDofs(ess_bdr, dofs_list, 1);
      ess_tdofs.Append(dofs_list);
      H1FESpace.GetEssentialVDofs(ess_bdr, dofs_marker, 1);
      FiniteElementSpace::MarkerToList(dofs_marker, dofs_list);
      ess_vdofs.Append(dofs_list);
   }

   // Define the explicit ODE solver used for time integration.
   ODESolver *ode_solver = NULL;
   switch (ode_solver_type)
   {
      case 1: ode_solver = new ForwardEulerSolver; break;
      case 2: ode_solver = new RK2Solver(0.5); break;
      case 3: ode_solver = new RK3SSPSolver; break;
      case 4: ode_solver = new RK4Solver; break;
      case 6: ode_solver = new RK6Solver; break;
      case 7: ode_solver = new RK2AvgSolver; break;
      default:
         if (myid == 0)
         {
            cout << "Unknown ODE solver type: " << ode_solver_type << '\n';
         }
         delete pmesh;
         MPI_Finalize();
         return 3;
   }

   const HYPRE_Int glob_size_l2 = L2FESpace.GlobalTrueVSize();
   const HYPRE_Int glob_size_h1 = H1FESpace.GlobalTrueVSize();
   if (mpi.Root())
   {
      cout << "Number of kinematic (position, velocity) dofs: "
           << glob_size_h1 << endl;
      cout << "Number of specific internal energy dofs: "
           << glob_size_l2 << endl;
   }

   // The monolithic BlockVector stores unknown fields as:
   // - 0 -> position
   // - 1 -> velocity
   // - 2 -> specific internal energy
   // - 3 -> stress
   // - 4 -> plastic strain
   const int Vsize_l2 = L2FESpace.GetVSize();
   const int Vsize_h1 = H1FESpace.GetVSize();
   // Array<int> offset(5);
   Array<int> offset(6); // when you change this number, you should chnage block offset in solver.cpp too
   offset[0] = 0;
   offset[1] = offset[0] + Vsize_h1;
   offset[2] = offset[1] + Vsize_h1;
   offset[3] = offset[2] + Vsize_l2;
   offset[4] = offset[3] + Vsize_l2*3*(dim-1);
   offset[5] = offset[4] + Vsize_l2;
   BlockVector S(offset, Device::GetMemoryType());

   // Define GridFunction objects for the position, velocity and specific
   // internal energy. There is no function for the density, as we can always
   // compute the density values given the current mesh position, using the
   // property of pointwise mass conservation.
   ParGridFunction x_gf, v_gf, e_gf, s_gf;
   ParGridFunction p_gf;
   x_gf.MakeRef(&H1FESpace, S, offset[0]);
   v_gf.MakeRef(&H1FESpace, S, offset[1]);
   e_gf.MakeRef(&L2FESpace, S, offset[2]);
   s_gf.MakeRef(&L2FESpace_stress, S, offset[3]);
   p_gf.MakeRef(&L2FESpace, S, offset[4]);

   // Initialize x_gf using the starting mesh coordinates.
   pmesh->SetNodalGridFunction(&x_gf);
   // Sync the data location of x_gf with its base, S
   x_gf.SyncAliasMemory(S);
   
   // Initialize the velocity.
   VectorFunctionCoefficient v_coeff(pmesh->Dimension(), v0);
   v_gf.ProjectCoefficient(v_coeff);

   // For the Sedov test, we use a delta function at the origin.
   // Vector dir(pmesh->Dimension());
   // L2_FECollection h1_fec(order_v, pmesh->Dimension());
   // ParFiniteElementSpace h1_fes(pmesh, &h1_fec);
   // ParGridFunction h1_v(&h1_fes);
   // dir=0.0;
   // dir(1)=1.0;
   // // VectorDeltaCoefficient v2_coeff(pmesh->Dimension());
   // // v2_coeff.SetScale(blast_energy2);
   // // v2_coeff.SetDeltaCenter(cent);
   // // v2_coeff.SetDirection(dir);
   // VectorDeltaCoefficient v2_coeff(dir, blast_position2[0], blast_position2[1], blast_energy2);
  
   // h1_v.ProjectCoefficient(v2_coeff);
   // // v_gf.ProjectGridFunction(h1_v);

   // Sync the data location of v_gf with its base, S
   v_gf.SyncAliasMemory(S);

   // Initialize density and specific internal energy values. We interpolate in
   // a non-positive basis to get the correct values at the dofs. Then we do an
   // L2 projection to the positive basis in which we actually compute. The goal
   // is to get a high-order representation of the initial condition. Note that
   // this density is a temporary function and it will not be updated during the
   // time evolution.
   Vector z_rho(pmesh->attributes.Max());
   z_rho = 2800;
   Vector s_rho(pmesh->attributes.Max());
   s_rho = 2800*mscale;

   ParGridFunction rho0_gf(&L2FESpace);
   PWConstCoefficient rho0_coeff(z_rho);
   PWConstCoefficient scale_rho0_coeff(s_rho);
   L2_FECollection l2_fec(order_e, pmesh->Dimension());
   ParFiniteElementSpace l2_fes(pmesh, &l2_fec);
   ParGridFunction l2_rho0_gf(&l2_fes), l2_e(&l2_fes);
   l2_rho0_gf.ProjectCoefficient(rho0_coeff);
   rho0_gf.ProjectGridFunction(l2_rho0_gf);

   /*
   ParGridFunction rho0_gf(&L2FESpace);
   FunctionCoefficient rho0_coeff(rho0);
   // FunctionCoefficient super_rho0_coeff(rho0);
   L2_FECollection l2_fec(order_e, pmesh->Dimension());
   ParFiniteElementSpace l2_fes(pmesh, &l2_fec);
   ParGridFunction l2_rho0_gf(&l2_fes), l2_e(&l2_fes);
   l2_rho0_gf.ProjectCoefficient(rho0_coeff);
   rho0_gf.ProjectGridFunction(l2_rho0_gf);
   */
   
   if (problem == 1)
   {
      // For the Sedov test, we use a delta function at the origin.
      DeltaCoefficient e_coeff(blast_position[0], blast_position[1],
                               blast_position[2], blast_energy);
      l2_e.ProjectCoefficient(e_coeff);
   }
   else
   {
      FunctionCoefficient e_coeff(e0);
      l2_e.ProjectCoefficient(e_coeff);
   }
   e_gf.ProjectGridFunction(l2_e);
   e_gf = 0;
   // Sync the data location of e_gf with its base, S
   e_gf.SyncAliasMemory(S);

   // Piecewise constant elastic stiffness over the Lagrangian mesh.
   // Lambda and Mu is Lame's first and second constants
   Vector lambda(pmesh->attributes.Max());
   lambda = 200e6-200e6*(2.0/3.0);
   // lambda = 200.0e6;
   PWConstCoefficient lambda_func(lambda);
   Vector mu(pmesh->attributes.Max());
   mu = 200.0e6;
   PWConstCoefficient mu_func(mu);
   
   // Project PWConstCoefficient to grid function
   L2_FECollection lambda_fec(order_e, pmesh->Dimension());
   ParFiniteElementSpace lambda_fes(pmesh, &lambda_fec);
   ParGridFunction lambda0_gf(&lambda_fes);
   lambda0_gf.ProjectCoefficient(lambda_func);
   
   L2_FECollection mu_fec(order_e, pmesh->Dimension());
   ParFiniteElementSpace mu_fes(pmesh, &mu_fec);
   ParGridFunction mu0_gf(&mu_fes);
   mu0_gf.ProjectCoefficient(mu_func);

   // Piecewise constant for material index
   Vector mat(pmesh->attributes.Max());
   for (int i = 0; i < pmesh->attributes.Max(); i++)
   {
      mat[i] = i;
   }
   PWConstCoefficient mat_func(mat);

   // Project PWConstCoefficient to grid function
   L2_FECollection mat_fec(order_e, pmesh->Dimension());
   ParFiniteElementSpace mat_fes(pmesh, &mat_fec);
   ParGridFunction mat_gf(&mat_fes);
   mat_gf.ProjectCoefficient(mat_func);

   // Material properties of Plasticity
   Vector tension_cutoff(pmesh->attributes.Max());
   Vector cohesion(pmesh->attributes.Max());
   Vector friction_angle(pmesh->attributes.Max());
   Vector dilation_angle(pmesh->attributes.Max());
   tension_cutoff = 0.0;
   cohesion = 1.0e6;
   friction_angle = 10.0;
   dilation_angle = 10.0;

   StressCoefficient stress_coef(dim, v_gf, lambda_func, mu_func);
   s_gf.ProjectCoefficient(stress_coef);
   s_gf=0.0;
   s_gf.SyncAliasMemory(S);

   // Initializing plastic strain (J2 strain invariant)
   FunctionCoefficient p_coeff(p0);
   p_gf.ProjectCoefficient(p_coeff);
   p_gf.SyncAliasMemory(S);

   ParGridFunction u_gf(&H1FESpace);  // Displacment
   ParGridFunction x0_gf(&H1FESpace); // Initial mesh (reference configuration)
   // Initialize x_gf using the starting mesh coordinates.
   pmesh->SetNodalGridFunction(&x0_gf);
   u_gf = 0.0; 

   // L2_FECollection mat_fec(0, pmesh->Dimension());
   // ParFiniteElementSpace mat_fes(pmesh, &mat_fec);
   // ParGridFunction mat_gf(&mat_fes);
   // FunctionCoefficient mat_coeff(gamma_func);
   // mat_gf.ProjectCoefficient(mat_coeff);
     
   /*
   // Initialize stress tensor
   L2_FECollection sig_fec(order_e, pmesh->Dimension());
   // ParFiniteElementSpace sig_fespace(pmesh, &sig_fec);
   ParFiniteElementSpace sig_fespace(pmesh, &sig_fec, dim*dim);
   ParGridFunction sigma_gf(&sig_fespace); 
   StressCoefficient stress_coef(dim, v_gf, lambda_func, mu_func);
   sigma_gf.ProjectCoefficient(stress_coef);
   */

   // Additional details, depending on the problem.
   /*
   DenseMatrix esig(dim);
   DenseMatrix plastic_sig(dim);
   DenseMatrix plastic_str(dim);
   esig=0.0; plastic_sig=0.0; plastic_str=0.0;
   double eig_sig_var[3], eig_sig_vec[9];

   double sig1{0.0};
   double sig3{0.0};

   double fs{0.0};
   double ft{0.0};
   double fh{0.0};
   double N_phi{0.0};
   double st_N_phi{0.0};
   double N_psi{0.0};
   double beta{0.0};
   */
   
   int source = 0; bool visc = false, vorticity = false;
   switch (problem)
   {
      case 0: if (pmesh->Dimension() == 2) { source = 1; } visc = false; break;
      case 1: visc = true; break;
      case 2: visc = true; break;
      case 3: visc = true; S.HostRead(); break;
      case 4: visc = false; break;
      case 5: visc = true; break;
      case 6: visc = true; break;
      case 7: source = 2; visc = true; vorticity = true;  break;
      default: MFEM_ABORT("Wrong problem specification!");
   }
   if (impose_visc) { visc = true; }

   geodynamics::LagrangianGeoOperator geo(S.Size(),
                                                H1FESpace, L2FESpace, L2FESpace_stress, ess_tdofs,
                                                rho0_coeff, scale_rho0_coeff, rho0_gf,
                                                mat_gf, source, cfl,
                                                visc, vorticity, p_assembly,
                                                cg_tol, cg_max_iter, ftz_tol,
                                                order_q, lambda0_gf, mu0_gf, mscale, gravity, 
                                                lambda, mu, tension_cutoff, cohesion, friction_angle, dilation_angle);

   socketstream vis_rho, vis_v, vis_e;
   char vishost[] = "localhost";
   int  visport   = 19916;

   ParGridFunction rho_gf;
   if (visualization || visit || paraview) { geo.ComputeDensity(rho_gf); }
   const double energy_init = geo.InternalEnergy(e_gf) +
                              geo.KineticEnergy(v_gf);

   if (visualization)
   {
      // Make sure all MPI ranks have sent their 'v' solution before initiating
      // another set of GLVis connections (one from each rank):
      MPI_Barrier(pmesh->GetComm());
      vis_rho.precision(8);
      vis_v.precision(8);
      vis_e.precision(8);
      int Wx = 0, Wy = 0; // window position
      const int Ww = 350, Wh = 350; // window size
      int offx = Ww+10; // window offsets
      if (problem != 0 && problem != 4)
      {
         geodynamics::VisualizeField(vis_rho, vishost, visport, rho_gf,
                                       "Density", Wx, Wy, Ww, Wh);
      }
      Wx += offx;
      geodynamics::VisualizeField(vis_v, vishost, visport, v_gf,
                                    "Velocity", Wx, Wy, Ww, Wh);
      Wx += offx;
      geodynamics::VisualizeField(vis_e, vishost, visport, e_gf,
                                    "Specific Internal Energy", Wx, Wy, Ww, Wh);
   }

   // Save data for VisIt visualization.
   VisItDataCollection visit_dc(basename, pmesh);
   if (visit)
   {
      visit_dc.RegisterField("Density",  &rho_gf);
      visit_dc.RegisterField("Velocity", &v_gf);
      visit_dc.RegisterField("Specific Internal Energy", &e_gf);
      // visit_dc.RegisterField("stress xx", &mu_gf);
      // visit_dc.RegisterField("stress yy", &lambda_gf);
      visit_dc.SetCycle(0);
      visit_dc.SetTime(0.0);
      visit_dc.Save();
   }

   ParaViewDataCollection *pd = NULL;
   if (paraview)
   {
      pd = new ParaViewDataCollection(basename, pmesh);
      pd->SetPrefixPath("ParaView");
      pd->RegisterField("Density",  &rho_gf);
      pd->RegisterField("Displacement", &u_gf);
      pd->RegisterField("Velocity", &v_gf);
      pd->RegisterField("Specific Internal Energy", &e_gf);
      pd->RegisterField("Stress", &s_gf);
      pd->RegisterField("Plastic Strain", &p_gf);
      // pd->SetLevelsOfDetail(order);
      // pd->SetDataFormat(VTKFormat::BINARY);
      pd->SetDataFormat(VTKFormat::ASCII);
      pd->SetHighOrderOutput(true);
      pd->SetCycle(0);
      pd->SetTime(0.0);
      pd->Save();
   }

   // Perform time-integration (looping over the time iterations, ti, with a
   // time-step dt). The object oper is of type LagrangianGeoOperator that
   // defines the Mult() method that used by the time integrators.
   ode_solver->Init(geo);
   geo.ResetTimeStepEstimate();
   double t = 0.0, dt = 0.0, t_old;
   dt = geo.GetTimeStepEstimate(S, dt); // To provide dt before the estimate, initializing is necessary
   
   // dt = init_dt;
   bool last_step = false;
   int steps = 0;
   BlockVector S_old(S);
   long mem=0, mmax=0, msum=0;
   int checks = 0;
   //   const double internal_energy = geo.InternalEnergy(e_gf);
   //   const double kinetic_energy = geo.KineticEnergy(v_gf);
   //   if (mpi.Root())
   //   {
   //      cout << std::fixed;
   //      cout << "step " << std::setw(5) << 0
   //            << ",\tt = " << std::setw(5) << std::setprecision(4) << t
   //            << ",\tdt = " << std::setw(5) << std::setprecision(6) << dt
   //            << ",\t|IE| = " << std::setprecision(10) << std::scientific
   //            << internal_energy
   //            << ",\t|KE| = " << std::setprecision(10) << std::scientific
   //            << kinetic_energy
   //            << ",\t|E| = " << std::setprecision(10) << std::scientific
   //            << kinetic_energy+internal_energy;
   //      cout << std::fixed;
   //      if (mem_usage)
   //      {
   //         cout << ", mem: " << mmax << "/" << msum << " MB";
   //      }
   //      cout << endl;
   //   }

   if (mpi.Root()) 
   {
      std::cout<<""<<std::endl;
      std::cout<<"simulation starts"<<std::endl;
   }

   for (int ti = 1; !last_step; ti++)
   {
      if (t + dt >= t_final)
      {
         dt = t_final - t;
         last_step = true;
      }
      if (steps == max_tsteps) { last_step = true; }
      S_old = S;
      t_old = t;
      geo.ResetTimeStepEstimate();

      // S is the vector of dofs, t is the current time, and dt is the time step
      // to advance.
      ode_solver->Step(S, t, dt);
      steps++;

      // Adaptive time step control.
      const double dt_est = geo.GetTimeStepEstimate(S, dt);
      // const double dt_est = geo.GetTimeStepEstimate(S, dt, mpi.Root());
      // const double dt_est = geo.GetTimeStepEstimate(S);
      if (dt_est < dt)
      {
         // Repeat (solve again) with a decreased time step - decrease of the
         // time estimate suggests appearance of oscillations.
         dt *= 0.85;
         if (dt < std::numeric_limits<double>::epsilon())
         { MFEM_ABORT("The time step crashed!"); }
         t = t_old;
         S = S_old;
         geo.ResetQuadratureData();
         // if (mpi.Root()) { cout << "Repeating step " << ti << endl; }
         if (steps < max_tsteps) { last_step = false; }
         ti--; continue;
      }
      else if (dt_est > 1.25 * dt) { dt *= 1.02; }

      // Ensure the sub-vectors x_gf, v_gf, and e_gf know the location of the
      // data in S. This operation simply updates the Memory validity flags of
      // the sub-vectors to match those of S.
      x_gf.SyncAliasMemory(S);
      v_gf.SyncAliasMemory(S);
      e_gf.SyncAliasMemory(S);
      s_gf.SyncAliasMemory(S);
      p_gf.SyncAliasMemory(S);

      // Returnmapping (s_gf, p_gf, mat_gf, dim, lambda, mu, tension_cutoff, cohesion, friction_angle, dilation_angle);
      s_gf.SyncAliasMemory(S);

      // for( int i = 0; i < Vsize_l2; i++ )
      // {  
      //    esig=0.0; plastic_sig=0.0; plastic_str=0.0;
      //    double eig_sig_var[3], eig_sig_vec[9];
      //    esig(0,0) = s_gf[i+Vsize_l2*0]; esig(0,1) = s_gf[i+Vsize_l2*3]; esig(0,2) = s_gf[i+Vsize_l2*4]; //
      //    esig(1,0) = s_gf[i+Vsize_l2*3]; esig(1,1) = s_gf[i+Vsize_l2*1]; esig(1,2) = s_gf[i+Vsize_l2*5];
      //    esig(2,0) = s_gf[i+Vsize_l2*4]; esig(2,1) = s_gf[i+Vsize_l2*5]; esig(2,2) = s_gf[i+Vsize_l2*2];

      //    esig.CalcEigenvalues(eig_sig_var, eig_sig_vec);

      //    Vector sig_var(eig_sig_var, dim);
      //    Vector sig_dir(eig_sig_vec, dim);

      //    // sig1 = sig_var.Min(); // most compressive 
      //    // sig3 = sig_var.Max(); // least compressive

      //    auto max_it = std::max_element(sig_var.begin(), sig_var.end()); // find iterator to max element
      //    auto min_it = std::min_element(sig_var.begin(), sig_var.end()); // find iterator to max element
         
      //    int max_index = std::distance(sig_var.begin(), max_it); // calculate index of max element
      //    int min_index = std::distance(sig_var.begin(), min_it); // calculate index of max element
      //    int itm_index = 0; // calculate index of max element

      //    if (max_index + min_index == 1) {itm_index = 2;}
      //    else if(max_index + min_index == 2) {itm_index = 1;}
      //    else {itm_index = 0;}

      //    sig1 = sig_var[min_index];
      //    sig3 = sig_var[max_index];

      //    N_phi = (1+sin(M_PI*friction_angle[0]/180.0))/(1-sin(M_PI*friction_angle[0]/180.0));
      //    st_N_phi = cos(M_PI*friction_angle[0]/180.0)/(1-sin(M_PI*friction_angle[0]/180.0));
      //    N_psi = -1*(1+sin(M_PI*dilation_angle[0]/180.0))/(1-sin(M_PI*dilation_angle[0]/180.0));
      //    // shear failure function
      //    fs = sig1 - N_phi*sig3 + 2*cohesion[0]*st_N_phi;
      //    // tension failure function
      //    ft = sig3 - (cohesion[0]/tan(M_PI*friction_angle[i]/180.0));
      //    // bisects the obtuse angle made by two yield function
      //    fh = sig3 - tension_cutoff[0] + (sqrt(N_phi*N_phi + 1.0)+ N_phi)*(sig1 - N_phi*tension_cutoff[0] + 2*cohesion[0]*st_N_phi);

      //    if(fs < 0 & fh < 0)
      //    {
      //       beta = fs;
      //       beta = beta / (((lambda[0]+2*mu[0])*1 - N_phi*lambda[0]*1) + (lambda[0]*N_psi - N_phi*(lambda[0]+2*mu[0])*N_psi));

      //       if(dim == 2)
      //       {
      //          plastic_str(0,0) = (lambda[0]+2*mu[0])*beta * 1; plastic_str(1,1) = (lambda[0]+2*mu[0]) * beta * N_psi;
      //       }
      //       else
      //       {
      //          plastic_str(0,0) = (lambda[0]+2*mu[0] + lambda[0]*N_psi) * beta; 
      //          plastic_str(1,1) = (lambda[0] + lambda[0]*N_psi) * beta;
      //          plastic_str(2,2) = (lambda[0] + (lambda[0]+2*mu[0])*N_psi) * beta;

      //          // plastic_str(0,0) = (lambda[0]+2*mu[0]) * beta * 1; 
      //          // plastic_str(2,2) = (lambda[0]+2*mu[0]) * beta * N_psi;
      //       }
      //    }
      //    else if (ft > 0 & fh > 0)
      //    {
      //       beta = ft;
      //       if(dim == 2)
      //       {
      //          plastic_str(1,1) = (lambda[0]+2*mu[0]) * beta * 1;
      //       }
      //       else
      //       {
      //          plastic_str(2,2) = (lambda[0]+2*mu[0]) * beta * 1;
      //       }
      //    }


      //    // Rotating Principal axis to XYZ axis
      //    if(dim == 2)
      //    {
      //       plastic_sig(0,0) = ((sig_var[0]-plastic_str(0,0))*sig_dir[0]*sig_dir[0] + (sig_var[1]-plastic_str(1,1))*sig_dir[2]*sig_dir[2]);
      //       plastic_sig(0,1) = ((sig_var[0]-plastic_str(0,0))*sig_dir[0]*sig_dir[1] + (sig_var[1]-plastic_str(1,1))*sig_dir[2]*sig_dir[3]);
      //       plastic_sig(1,0) = ((sig_var[0]-plastic_str(0,0))*sig_dir[1]*sig_dir[0] + (sig_var[1]-plastic_str(1,1))*sig_dir[3]*sig_dir[2]);
      //       plastic_sig(1,1) = ((sig_var[0]-plastic_str(0,0))*sig_dir[1]*sig_dir[1] + (sig_var[1]-plastic_str(1,1))*sig_dir[3]*sig_dir[3]);
      //    }
      //    else
      //    {
      //       plastic_sig(0,0) = ((sig_var[min_index]-plastic_str(0,0))*sig_dir[0+min_index*3]*sig_dir[0+min_index*3] + (sig_var[itm_index]-plastic_str(1,1))*sig_dir[0+itm_index*3]*sig_dir[0+itm_index*3] + (sig_var[max_index]-plastic_str(2,2))*sig_dir[0+max_index*3]*sig_dir[0+max_index*3]);
      //       plastic_sig(0,1) = ((sig_var[min_index]-plastic_str(0,0))*sig_dir[0+min_index*3]*sig_dir[1+min_index*3] + (sig_var[itm_index]-plastic_str(1,1))*sig_dir[0+itm_index*3]*sig_dir[1+itm_index*3] + (sig_var[max_index]-plastic_str(2,2))*sig_dir[0+max_index*3]*sig_dir[1+max_index*3]);
      //       plastic_sig(0,2) = ((sig_var[min_index]-plastic_str(0,0))*sig_dir[0+min_index*3]*sig_dir[2+min_index*3] + (sig_var[itm_index]-plastic_str(1,1))*sig_dir[0+itm_index*3]*sig_dir[2+itm_index*3] + (sig_var[max_index]-plastic_str(2,2))*sig_dir[0+max_index*3]*sig_dir[2+max_index*3]);
      //       plastic_sig(1,0) = ((sig_var[min_index]-plastic_str(0,0))*sig_dir[1+min_index*3]*sig_dir[0+min_index*3] + (sig_var[itm_index]-plastic_str(1,1))*sig_dir[1+itm_index*3]*sig_dir[0+itm_index*3] + (sig_var[max_index]-plastic_str(2,2))*sig_dir[1+max_index*3]*sig_dir[0+max_index*3]);
      //       plastic_sig(1,1) = ((sig_var[min_index]-plastic_str(0,0))*sig_dir[1+min_index*3]*sig_dir[1+min_index*3] + (sig_var[itm_index]-plastic_str(1,1))*sig_dir[1+itm_index*3]*sig_dir[1+itm_index*3] + (sig_var[max_index]-plastic_str(2,2))*sig_dir[1+max_index*3]*sig_dir[1+max_index*3]);
      //       plastic_sig(1,2) = ((sig_var[min_index]-plastic_str(0,0))*sig_dir[1+min_index*3]*sig_dir[2+min_index*3] + (sig_var[itm_index]-plastic_str(1,1))*sig_dir[1+itm_index*3]*sig_dir[2+itm_index*3] + (sig_var[max_index]-plastic_str(2,2))*sig_dir[1+max_index*3]*sig_dir[2+max_index*3]);
      //       plastic_sig(2,0) = ((sig_var[min_index]-plastic_str(0,0))*sig_dir[2+min_index*3]*sig_dir[0+min_index*3] + (sig_var[itm_index]-plastic_str(1,1))*sig_dir[2+itm_index*3]*sig_dir[0+itm_index*3] + (sig_var[max_index]-plastic_str(2,2))*sig_dir[2+max_index*3]*sig_dir[0+max_index*3]);
      //       plastic_sig(2,1) = ((sig_var[min_index]-plastic_str(0,0))*sig_dir[2+min_index*3]*sig_dir[1+min_index*3] + (sig_var[itm_index]-plastic_str(1,1))*sig_dir[2+itm_index*3]*sig_dir[1+itm_index*3] + (sig_var[max_index]-plastic_str(2,2))*sig_dir[2+max_index*3]*sig_dir[1+max_index*3]);
      //       plastic_sig(2,2) = ((sig_var[min_index]-plastic_str(0,0))*sig_dir[2+min_index*3]*sig_dir[2+min_index*3] + (sig_var[itm_index]-plastic_str(1,1))*sig_dir[2+itm_index*3]*sig_dir[2+itm_index*3] + (sig_var[max_index]-plastic_str(2,2))*sig_dir[2+max_index*3]*sig_dir[2+max_index*3]);
      //    }

      //    // esig.Add(-1.0, plastic_sig); // Applying plastic stress correction.

      //    s_gf[i+Vsize_l2*0]=plastic_sig(0,0); s_gf[i+Vsize_l2*3]=plastic_sig(0,1); s_gf[i+Vsize_l2*4]=plastic_sig(0,2); //
      //    s_gf[i+Vsize_l2*3]=plastic_sig(1,0); s_gf[i+Vsize_l2*1]=plastic_sig(1,1); s_gf[i+Vsize_l2*5]=plastic_sig(1,2);
      //    s_gf[i+Vsize_l2*4]=plastic_sig(2,0); s_gf[i+Vsize_l2*5]=plastic_sig(2,1); s_gf[i+Vsize_l2*2]=plastic_sig(2,2);

      // }

      

      // Adding stress increment to total stress and storing spin rate
      // Make sure that the mesh corresponds to the new solution state. This is
      // needed, because some time integrators use different S-type vectors
      // and the oper object might have redirected the mesh positions to those.
      pmesh->NewNodes(x_gf, false);
      u_gf = x0_gf;
      u_gf -= x_gf;
      u_gf.Neg();

      // geo.ComputeDensity(rho_gf);
      
      if (last_step || (ti % vis_steps) == 0)
      {
         double lnorm = e_gf * e_gf, norm;
         MPI_Allreduce(&lnorm, &norm, 1, MPI_DOUBLE, MPI_SUM, pmesh->GetComm());
         if (mem_usage)
         {
            mem = GetMaxRssMB();
            MPI_Reduce(&mem, &mmax, 1, MPI_LONG, MPI_MAX, 0, pmesh->GetComm());
            MPI_Reduce(&mem, &msum, 1, MPI_LONG, MPI_SUM, 0, pmesh->GetComm());
         }
         const double internal_energy = geo.InternalEnergy(e_gf);
         const double kinetic_energy = geo.KineticEnergy(v_gf);
         if(year)
         {
            if (mpi.Root())
            {
            const double sqrt_norm = sqrt(norm);

            cout << std::fixed;
            cout << "step " << std::setw(5) << ti
                 << ",\tt = " << std::setw(5) << std::setprecision(4) << t/86400/365.25
                 << ",\tdt = " << std::setw(5) << std::setprecision(6) << std::scientific << dt/86400/365.25
                 << ",\t|e| = " << std::setprecision(10) << std::scientific
                 << sqrt_norm;
            //  << ",\t|IE| = " << std::setprecision(10) << std::scientific
            //  << internal_energy
            //   << ",\t|KE| = " << std::setprecision(10) << std::scientific
            //  << kinetic_energy
            //   << ",\t|E| = " << std::setprecision(10) << std::scientific
            //  << kinetic_energy+internal_energy;
            cout << std::fixed;
            if (mem_usage)
               {
                  cout << ", mem: " << mmax << "/" << msum << " MB";
               }
            cout << endl;
            }
         }
         else
         {
            if (mpi.Root())
            {
            const double sqrt_norm = sqrt(norm);

            cout << std::fixed;
            cout << "step " << std::setw(5) << ti
                 << ",\tt = " << std::setw(5) << std::setprecision(4) << t
                 << ",\tdt = " << std::setw(5) << std::setprecision(6) << dt
                 << ",\t|e| = " << std::setprecision(10) << std::scientific
                 << sqrt_norm;
               //   << ",\t|IE| = " << std::setprecision(10) << std::scientific
               //   << internal_energy
               //   << ",\t|KE| = " << std::setprecision(10) << std::scientific
               //   << kinetic_energy;
            //   << ",\t|E| = " << std::setprecision(10) << std::scientific
            //  << kinetic_energy+internal_energy;
            cout << std::fixed;
            if (mem_usage)
               {
                  cout << ", mem: " << mmax << "/" << msum << " MB";
               }
            cout << endl;
            }
         }
         
         // Make sure all ranks have sent their 'v' solution before initiating
         // another set of GLVis connections (one from each rank):
         MPI_Barrier(pmesh->GetComm());

         if (visualization || visit || gfprint || paraview) { geo.ComputeDensity(rho_gf); }
         if (visualization)
         {
            int Wx = 0, Wy = 0; // window position
            int Ww = 350, Wh = 350; // window size
            int offx = Ww+10; // window offsets
            if (problem != 0 && problem != 4)
            {
               geodynamics::VisualizeField(vis_rho, vishost, visport, rho_gf,
                                             "Density", Wx, Wy, Ww, Wh);
            }
            Wx += offx;
            geodynamics::VisualizeField(vis_v, vishost, visport,
                                          v_gf, "Velocity", Wx, Wy, Ww, Wh);
            Wx += offx;
            geodynamics::VisualizeField(vis_e, vishost, visport, e_gf,
                                          "Specific Internal Energy",
                                          Wx, Wy, Ww,Wh);
            Wx += offx;
         }

         if (visit)
         {
            visit_dc.SetCycle(ti);
            visit_dc.SetTime(t);
            visit_dc.Save();
         }

         if (paraview)
         {
            pd->SetCycle(ti);
            pd->SetTime(t);
            pd->Save();
         }

         if (gfprint)
         {
            std::ostringstream mesh_name, rho_name, v_name, e_name;
            mesh_name << basename << "_" << ti << "_mesh";
            rho_name  << basename << "_" << ti << "_rho";
            v_name << basename << "_" << ti << "_v";
            e_name << basename << "_" << ti << "_e";

            std::ofstream mesh_ofs(mesh_name.str().c_str());
            mesh_ofs.precision(8);
            pmesh->PrintAsOne(mesh_ofs);
            mesh_ofs.close();

            std::ofstream rho_ofs(rho_name.str().c_str());
            rho_ofs.precision(8);
            rho_gf.SaveAsOne(rho_ofs);
            rho_ofs.close();

            std::ofstream v_ofs(v_name.str().c_str());
            v_ofs.precision(8);
            v_gf.SaveAsOne(v_ofs);
            v_ofs.close();

            std::ofstream e_ofs(e_name.str().c_str());
            e_ofs.precision(8);
            e_gf.SaveAsOne(e_ofs);
            e_ofs.close();
         }
      }

      // Problems checks
      if (check)
      {
         double lnorm = e_gf * e_gf, norm;
         MPI_Allreduce(&lnorm, &norm, 1, MPI_DOUBLE, MPI_SUM, pmesh->GetComm());
         const double e_norm = sqrt(norm);
         MFEM_VERIFY(rs_levels==0 && rp_levels==0, "check: rs, rp");
         MFEM_VERIFY(order_v==2, "check: order_v");
         MFEM_VERIFY(order_e==1, "check: order_e");
         MFEM_VERIFY(ode_solver_type==4, "check: ode_solver_type");
         MFEM_VERIFY(t_final == 0.6, "check: t_final");
         MFEM_VERIFY(cfl==0.5, "check: cfl");
         MFEM_VERIFY(strncmp(mesh_file, "default", 7) == 0, "check: mesh_file");
         MFEM_VERIFY(dim==2 || dim==3, "check: dimension");
         Checks(ti, e_norm, checks);
      }
   }
   MFEM_VERIFY(!check || checks == 2, "Check error!");

   switch (ode_solver_type)
   {
      case 2: steps *= 2; break;
      case 3: steps *= 3; break;
      case 4: steps *= 4; break;
      case 6: steps *= 6; break;
      case 7: steps *= 2;
   }

   geo.PrintTimingData(mpi.Root(), steps, fom);

   if (mem_usage)
   {
      mem = GetMaxRssMB();
      MPI_Reduce(&mem, &mmax, 1, MPI_LONG, MPI_MAX, 0, pmesh->GetComm());
      MPI_Reduce(&mem, &msum, 1, MPI_LONG, MPI_SUM, 0, pmesh->GetComm());
   }

   const double energy_final = geo.InternalEnergy(e_gf) +
                               geo.KineticEnergy(v_gf);
   if (mpi.Root())
   {
      cout << endl;
      cout << "Energy  diff: " << std::scientific << std::setprecision(2)
           << fabs(energy_init - energy_final) << endl;
      if (mem_usage)
      {
         cout << "Maximum memory resident set size: "
              << mmax << "/" << msum << " MB" << endl;
      }
   }

   // Print the error.
   // For problems 0 and 4 the exact velocity is constant in time.
   if (problem == 0 || problem == 4)
   {
      const double error_max = v_gf.ComputeMaxError(v_coeff),
                   error_l1  = v_gf.ComputeL1Error(v_coeff),
                   error_l2  = v_gf.ComputeL2Error(v_coeff);
      if (mpi.Root())
      {
         cout << "L_inf  error: " << error_max << endl
              << "L_1    error: " << error_l1 << endl
              << "L_2    error: " << error_l2 << endl;
      }
   }

   if (visualization)
   {
      vis_v.close();
      vis_e.close();
   }

   // Free the used memory.
   delete ode_solver;
   delete pmesh;

   return 0;
}

double rho0(const Vector &x)
{
   return 2800.0; // This case in initialized in main().
}

double gamma_func(const Vector &x)
{
   return 1.0; // This case in initialized in main().
}

static double rad(double x, double y) { return sqrt(x*x + y*y); }

void v0(const Vector &x, Vector &v)
{
   const double atn = dim!=1 ? pow((x(0)*(1.0-x(0))*4*x(1)*(1.0-x(1))*4.0),
                                   0.4) : 0.0;
   const double s = 0.1/64.;
   v = 0.0;
   if(x(0) == 1)
   {
      v(0)=-1e-5;
   } 
}

double e0(const Vector &x)
{
   return 0.0; // This case in initialized in main().
}

double p0(const Vector &x)
{
   return 0.0; // This case in initialized in main().
}

static void display_banner(std::ostream &os)
{
   os << endl
      << "       __                __                 " << endl
      << "      / /   ____  ____  / /_  ____  _____   " << endl
      << "     / /   / __ `/ __ `/ __ \\/ __ \\/ ___/ " << endl
      << "    / /___/ /_/ / /_/ / / / / /_/ (__  )    " << endl
      << "   /_____/\\__,_/\\__, /_/ /_/\\____/____/  " << endl
      << "               /____/                       " << endl << endl;
}

static long GetMaxRssMB()
{
   struct rusage usage;
   if (getrusage(RUSAGE_SELF, &usage)) { return -1; }
#ifndef __APPLE__
   const long unit = 1024; // kilo
#else
   const long unit = 1024*1024; // mega
#endif
   return usage.ru_maxrss/unit; // mega bytes
}

static void Checks(const int ti, const double nrm, int &chk)
{
   const double eps = 1.e-13;
   //printf("\033[33m%.15e\033[m\n",nrm);

   auto check = [&](int p, int i, const double res)
   {
      auto rerr = [](const double a, const double v, const double eps)
      {
         MFEM_VERIFY(fabs(a) > eps && fabs(v) > eps, "One value is near zero!");
         const double err_a = fabs((a-v)/a);
         const double err_v = fabs((a-v)/v);
         return fmax(err_a, err_v) < eps;
      };
      if (problem == p && ti == i)
      { chk++; MFEM_VERIFY(rerr(nrm, res, eps), "P"<<problem<<", #"<<i); }
   };

   const double it_norms[2][8][2][2] = // dim, problem, {it,norm}
   {
      {
         {{5, 6.546538624534384e+00}, { 27, 7.588576357792927e+00}},
         {{5, 3.508254945225794e+00}, { 15, 2.756444596823211e+00}},
         {{5, 1.020745795651244e+01}, { 59, 1.721590205901898e+01}},
         {{5, 8.000000000000000e+00}, { 16, 8.000000000000000e+00}},
         {{5, 3.446324942352448e+01}, { 18, 3.446844033767240e+01}},
         {{5, 1.030899557252528e+01}, { 36, 1.057362418574309e+01}},
         {{5, 8.039707010835693e+00}, { 36, 8.316970976817373e+00}},
         {{5, 1.514929259650760e+01}, { 25, 1.514931278155159e+01}},
      },
      {
         {{5, 1.198510951452527e+03}, {188, 1.199384410059154e+03}},
         {{5, 1.339163718592566e+01}, { 28, 7.521073677397994e+00}},
         {{5, 2.041491591302486e+01}, { 59, 3.443180411803796e+01}},
         {{5, 1.600000000000000e+01}, { 16, 1.600000000000000e+01}},
         {{5, 6.892649884704898e+01}, { 18, 6.893688067534482e+01}},
         {{5, 2.061984481890964e+01}, { 36, 2.114519664792607e+01}},
         {{5, 1.607988713996459e+01}, { 36, 1.662736010353023e+01}},
         {{5, 3.029858112572883e+01}, { 24, 3.029858832743707e+01}}
      }
   };

   for (int p=0; p<8; p++)
   {
      for (int i=0; i<2; i++)
      {
         const int it = it_norms[dim-2][p][i][0];
         const double norm = it_norms[dim-2][p][i][1];
         check(p, it, norm);
      }
   }
}

void Returnmapping (Vector &s_gf, Vector &p_gf, Vector &mat_gf, int &dim, Vector &lambda, Vector &mu, Vector &tension_cutoff, Vector &cohesion, Vector &friction_angle, Vector &dilation_angle) 
{
   DenseMatrix esig(3);
   DenseMatrix plastic_sig(3);
   DenseMatrix plastic_str(3);
   esig=0.0; plastic_sig=0.0; plastic_str=0.0;
   double eig_sig_var[3], eig_sig_vec[9];

   double sig1{0.0};
   double sig3{0.0};
   double  syy{0.0}; // Syy is non-zero value in plane strain condition
   double msig{0.0}; // mean stress
   double evol{0.0}; // volumetric strain
   double DEG2RAD{M_PI/180.0};
   double depls{0.0}; // 2nd invariant of plastic strain

   double fs{0.0};
   double ft{0.0};
   double fh{0.0};
   double N_phi{0.0};
   double st_N_phi{0.0};
   double N_psi{0.0};
   double beta{0.0};
   int mat{0};
   int nsize{mat_gf.Size()};

   for( int i = 0; i < nsize; i++ )
   {  
         esig=0.0; plastic_sig=0.0; plastic_str=0.0;
         double eig_sig_var[3], eig_sig_vec[9];

         mat = mat_gf[i];

          if(dim ==2) 
         {
            // 2D in plane strain condition
            // sxx, syy, szz, sxz are non zeros.
            // sxy, syz are zero.
            msig = (s_gf[i+nsize*0] + s_gf[i+nsize*1])*0.5;
            evol = msig/(lambda[mat]+mu[mat]);
            syy  = evol * lambda[mat];
            esig(0,0) = s_gf[i+nsize*0]; esig(0,1) = 0.0; esig(0,2) = s_gf[i+nsize*2]; 
            esig(1,0) =             0,0; esig(1,1) = syy; esig(1,2) =             0.0;
            esig(2,0) = s_gf[i+nsize*2]; esig(2,1) = 0.0; esig(2,2) = s_gf[i+nsize*1];
         }
         else
         {
            esig(0,0) = s_gf[i+nsize*0]; esig(0,1) = s_gf[i+nsize*3]; esig(0,2) = s_gf[i+nsize*4]; 
            esig(1,0) = s_gf[i+nsize*3]; esig(1,1) = s_gf[i+nsize*1]; esig(1,2) = s_gf[i+nsize*5];
            esig(2,0) = s_gf[i+nsize*4]; esig(2,1) = s_gf[i+nsize*5]; esig(2,2) = s_gf[i+nsize*2];
         }
         
         esig.CalcEigenvalues(eig_sig_var, eig_sig_vec); 

         Vector sig_var(eig_sig_var, 3);
         Vector sig_dir(eig_sig_vec, 3);

         auto max_it = std::max_element(sig_var.begin(), sig_var.end()); // find iterator to max element
         auto min_it = std::min_element(sig_var.begin(), sig_var.end()); // find iterator to min element
         
         int max_index = std::distance(sig_var.begin(), max_it); // calculate index of max element
         int min_index = std::distance(sig_var.begin(), min_it); // calculate index of min element
         
         int itm_index = 0; // calculate index of intermediate element
         if (max_index + min_index == 1) {itm_index = 2;}
         else if(max_index + min_index == 2) {itm_index = 1;}
         else {itm_index = 0;}

         sig1 = sig_var[min_index]; // most compressive pincipal stress
         sig3 = sig_var[max_index]; // least compressive pincipal stress

         N_phi = (1+sin(DEG2RAD*friction_angle[mat]))/(1-sin(DEG2RAD*friction_angle[mat]));
         st_N_phi = cos(DEG2RAD*friction_angle[mat])/(1-sin(DEG2RAD*friction_angle[mat]));
         N_psi = -1*(1+sin(DEG2RAD*dilation_angle[mat]))/(1-sin(DEG2RAD*dilation_angle[mat]));
         // shear failure function
         fs = sig1 - N_phi*sig3 + 2*cohesion[mat]*st_N_phi;
         // tension failure function
         ft = sig3 - (cohesion[mat]/tan(DEG2RAD*friction_angle[i]));
         // bisects the obtuse angle made by two yield function
         fh = sig3 - tension_cutoff[mat] + (sqrt(N_phi*N_phi + 1.0)+ N_phi)*(sig1 - N_phi*tension_cutoff[mat] + 2*cohesion[mat]*st_N_phi);

         if(fs < 0 & fh < 0) // stress correction at shear failure
         {
            beta = fs;
            beta = beta / (((lambda[mat]+2*mu[mat])*1 - N_phi*lambda[mat]*1) + (lambda[mat]*N_psi - N_phi*(lambda[mat]+2*mu[mat])*N_psi));

            plastic_str(0,0) = (lambda[mat] + 2*mu[mat] + lambda[mat]*N_psi) * beta; 
            plastic_str(1,1) = (lambda[mat] + lambda[mat]*N_psi) * beta;
            plastic_str(2,2) = (lambda[mat] + (lambda[mat]+2*mu[mat])*N_psi) * beta;

            // reduced form of 2nd invariant
            if(dim ==2)
            {
               depls = std::fabs(beta) * std::sqrt((3 - 2*N_psi + 3*N_psi*N_psi) / 8); 
            }
            else
            {
               depls = std::fabs(beta) * std::sqrt((7 - 4*N_psi + 7*N_psi*N_psi) / 18);
            }
            
         }
         else if (ft > 0 & fh > 0) // stress correction at tension failure
         {
            beta = ft;
            plastic_str(0,0) = (lambda[mat]+2*mu[mat]) * beta * 1;
            plastic_str(1,1) = (lambda[mat]+2*mu[mat]) * beta * 1;
            plastic_str(2,2) = lambda[mat] * beta * 1;

            // reduced form of 2nd invariant
            if(dim ==2)
            {
               depls = std::fabs(beta) * std::sqrt(3. / 8);
            }
            else
            {
               depls = std::fabs(beta) * std::sqrt(7. / 18);
            }
            
         }

         // Rotating Principal axis to XYZ axis
         plastic_sig(0,0) = ((sig_var[min_index]-plastic_str(0,0))*sig_dir[0+min_index*3]*sig_dir[0+min_index*3] + (sig_var[itm_index]-plastic_str(1,1))*sig_dir[0+itm_index*3]*sig_dir[0+itm_index*3] + (sig_var[max_index]-plastic_str(2,2))*sig_dir[0+max_index*3]*sig_dir[0+max_index*3]);
         plastic_sig(0,1) = ((sig_var[min_index]-plastic_str(0,0))*sig_dir[0+min_index*3]*sig_dir[1+min_index*3] + (sig_var[itm_index]-plastic_str(1,1))*sig_dir[0+itm_index*3]*sig_dir[1+itm_index*3] + (sig_var[max_index]-plastic_str(2,2))*sig_dir[0+max_index*3]*sig_dir[1+max_index*3]);
         plastic_sig(0,2) = ((sig_var[min_index]-plastic_str(0,0))*sig_dir[0+min_index*3]*sig_dir[2+min_index*3] + (sig_var[itm_index]-plastic_str(1,1))*sig_dir[0+itm_index*3]*sig_dir[2+itm_index*3] + (sig_var[max_index]-plastic_str(2,2))*sig_dir[0+max_index*3]*sig_dir[2+max_index*3]);
         plastic_sig(1,0) = ((sig_var[min_index]-plastic_str(0,0))*sig_dir[1+min_index*3]*sig_dir[0+min_index*3] + (sig_var[itm_index]-plastic_str(1,1))*sig_dir[1+itm_index*3]*sig_dir[0+itm_index*3] + (sig_var[max_index]-plastic_str(2,2))*sig_dir[1+max_index*3]*sig_dir[0+max_index*3]);
         plastic_sig(1,1) = ((sig_var[min_index]-plastic_str(0,0))*sig_dir[1+min_index*3]*sig_dir[1+min_index*3] + (sig_var[itm_index]-plastic_str(1,1))*sig_dir[1+itm_index*3]*sig_dir[1+itm_index*3] + (sig_var[max_index]-plastic_str(2,2))*sig_dir[1+max_index*3]*sig_dir[1+max_index*3]);
         plastic_sig(1,2) = ((sig_var[min_index]-plastic_str(0,0))*sig_dir[1+min_index*3]*sig_dir[2+min_index*3] + (sig_var[itm_index]-plastic_str(1,1))*sig_dir[1+itm_index*3]*sig_dir[2+itm_index*3] + (sig_var[max_index]-plastic_str(2,2))*sig_dir[1+max_index*3]*sig_dir[2+max_index*3]);
         plastic_sig(2,0) = ((sig_var[min_index]-plastic_str(0,0))*sig_dir[2+min_index*3]*sig_dir[0+min_index*3] + (sig_var[itm_index]-plastic_str(1,1))*sig_dir[2+itm_index*3]*sig_dir[0+itm_index*3] + (sig_var[max_index]-plastic_str(2,2))*sig_dir[2+max_index*3]*sig_dir[0+max_index*3]);
         plastic_sig(2,1) = ((sig_var[min_index]-plastic_str(0,0))*sig_dir[2+min_index*3]*sig_dir[1+min_index*3] + (sig_var[itm_index]-plastic_str(1,1))*sig_dir[2+itm_index*3]*sig_dir[1+itm_index*3] + (sig_var[max_index]-plastic_str(2,2))*sig_dir[2+max_index*3]*sig_dir[1+max_index*3]);
         plastic_sig(2,2) = ((sig_var[min_index]-plastic_str(0,0))*sig_dir[2+min_index*3]*sig_dir[2+min_index*3] + (sig_var[itm_index]-plastic_str(1,1))*sig_dir[2+itm_index*3]*sig_dir[2+itm_index*3] + (sig_var[max_index]-plastic_str(2,2))*sig_dir[2+max_index*3]*sig_dir[2+max_index*3]);

         // Updating new stress to grid function
         if(dim ==2)
         {
            s_gf[i+nsize*0]=plastic_sig(0,0); s_gf[i+nsize*2]=plastic_sig(0,2);
            s_gf[i+nsize*2]=plastic_sig(2,0); s_gf[i+nsize*1]=plastic_sig(2,2);
         }
         else
         {
            s_gf[i+nsize*0]=plastic_sig(0,0); s_gf[i+nsize*3]=plastic_sig(0,1); s_gf[i+nsize*4]=plastic_sig(0,2); 
            s_gf[i+nsize*3]=plastic_sig(1,0); s_gf[i+nsize*1]=plastic_sig(1,1); s_gf[i+nsize*5]=plastic_sig(1,2);
            s_gf[i+nsize*4]=plastic_sig(2,0); s_gf[i+nsize*5]=plastic_sig(2,1); s_gf[i+nsize*2]=plastic_sig(2,2);
         }

         // Adding 2nd invariant of plastic strain increment
         p_gf[i] += depls;
   }
}