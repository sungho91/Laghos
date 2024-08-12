#include <algorithm>  // For std::is_sorted
#include <cstdio>
#include <iostream>
#include <limits>
#include <sstream>

#include <boost/program_options.hpp>
namespace po = boost::program_options;

#include "parameters.hpp"
// #include "matprops.hpp"
// #include "utils.hpp"


static void declare_parameters(po::options_description &cfg,
                               Param &p)
{
    /* To have a new input parameter declared as such in parameters.hpp,
     *
     *     struct SectionType {
     *         type name;
     *     }
     *     struct Param {
     *         SectionType section;
     *     }
     *
     * add this line in this function:
     *
     *     ("section.name", po::value<type>(&p.section.name), "help string")
     *
     */

    cfg.add_options()
        ("sim.problem", po::value<int>(&p.sim.problem)->default_value(1),
         "Problem type: 1 to ??")
        ("sim.dim", po::value<int>(&p.sim.dim)->default_value(3),
         "Physical dimension of the problem: 2 or 3")
        ("sim.t_final", po::value<double>(&p.sim.t_final)->default_value(1.0),
         "Final time for the simulation in seconds")
        ("sim.max_tsteps", po::value<int>(&p.sim.max_tsteps)->default_value(-1),"Final time; start time is 0.") 
        ("sim.year", po::value<bool>(&p.sim.year)->default_value(false),
         "Is time unit year?")
        ("sim.visualization", po::value<bool>(&p.sim.visualization)->default_value(false),
         "Start in-app visualization? true or false") 
        ("sim.vis_steps", po::value<int>(&p.sim.vis_steps)->default_value(1000),
         "Visuaization output at every N step: N in integer")
        ("sim.visit", po::value<bool>(&p.sim.visit)->default_value(false),
         "Visualization outputs for VisIt? true or false")
        ("sim.paraview", po::value<bool>(&p.sim.paraview)->default_value(true),
         "Visualization outputs for ParaView? true or false")
        ("sim.gfprint", po::value<bool>(&p.sim.gfprint)->default_value(false),
         "Enable or disable result output (files in mfem format)") 
        ("sim.basename", po::value<std::string>(&p.sim.basename)->default_value("results/Laghost"),
         "Prefix for the output files")
        ("sim.device", po::value<std::string>(&p.sim.device)->default_value("cpu"),
         "Choose device, cpu or gpu")
        ("sim.dev", po::value<int>(&p.sim.dev)->default_value(0),
         "Choose a gpu device, 0, 1, ...")
        ("sim.check", po::value<bool>(&p.sim.check)->default_value(false),
         "Enable 2D checks.")
        ("sim.mem_usage", po::value<bool>(&p.sim.mem_usage)->default_value(false),
         "Print memory usage")
        ("sim.fom", po::value<bool>(&p.sim.fom)->default_value(false),
         "Enable figure of merit output.")
        ("sim.gpu_aware_mpi", po::value<bool>(&p.sim.gpu_aware_mpi)->default_value(false),
         "Enable GPU aware MPI communications.")
        ;

    cfg.add_options()
        ("solver.ode_solver_type", po::value<>(&p.solver.ode_solver_type)->default_value(7)," ")
        ("solver.cfl", po::value<double>(&p.solver.cfl)->default_value(0.5), " ")
        ("solver.cg_tol", po::value<double>(&p.solver.cg_tol)->default_value(1.0e-10)," ")
        ("solver.ftz_tol", po::value<double>(&p.solver.ftz_tol)->default_value(0.0)," ")
        ("solver.cg_max_iter", po::value<int>(&p.solver.cg_max_iter)->default_value(300)," ")
        ("solver.p_assembly", po::value<bool>(&p.solver.p_assembly)->default_value(false)," ")
        ("solver.impose_visc", po::value<bool>(&p.solver.impose_visc)->default_value(true)," ")
        ;

    cfg.add_options()
        ("bc.bc_unit", po::value<std::string>(&p.bc.bc_unit)->default_value("cm/yr"),"Unit of Velocity")
        ("bc.bc_ids", po::value<std::string>(&p.bc.bc_ids)->default_value("[0]"),"Boundary indicators '[d0, d1, d2, ...]")
        ("bc.bc_vxs", po::value<std::string>(&p.bc.bc_vxs)->default_value("[0]"), "Boundary veloicty x '[d0, d1, d2, ...]")
        ("bc.bc_vys", po::value<std::string>(&p.bc.bc_vys)->default_value("[0]"), "Boundary velocity y '[d0, d1, d2, ...]")
        ("bc.bc_vzs", po::value<std::string>(&p.bc.bc_vzs)->default_value("[0]"), "Boundary velocity z '[d0, d1, d2, ...]")
        ;

    cfg.add_options()
        ("control.pseudo_transient", po::value<bool>(&p.control.pseudo_transient)->default_value(false)," ")
        ("control.transient_num", po::value<int>(&p.control.transient_num)->default_value(5)," ")
        ("control.winkler_foundation", po::value<bool>(&p.control.winkler_foundation)->default_value(false)," ")
        ("control.winkler_flat", po::value<bool>(&p.control.winkler_flat)->default_value(false)," ")
        // ("control.flat_rate", po::value<double>(&p.control.flat_rate)->default_value(1.0e-7), " ")
        ("control.lithostatic", po::value<bool>(&p.control.lithostatic)->default_value(true)," ")
        ("control.init_dt", po::value<double>(&p.control.init_dt)->default_value(1.0), " ")
        ("control.mscale", po::value<double>(&p.control.mscale)->default_value(5.0e5), " ")
        ("control.gravity", po::value<double>(&p.control.gravity)->default_value(10.0), " ")
        ("control.thickness", po::value<double>(&p.control.thickness)->default_value(10.0e3), " ")
        ("control.winkler_rho", po::value<double>(&p.control.winkler_rho)->default_value(2700.0), " ")
        ("control.mass_bal", po::value<bool>(&p.control.mass_bal)->default_value(false)," ")
        ("control.dyn_damping", po::value<bool>(&p.control.dyn_damping)->default_value(true)," ")
        ("control.dyn_factor", po::value<double>(&p.control.dyn_factor)->default_value(0.8), " ")
        ("control.surf_proc", po::value<bool>(&p.control.surf_proc)->default_value(true)," ")
        ("control.surf_diff", po::value<double>(&p.control.surf_diff)->default_value(1.0e-7), " ")
        ("control.bott_proc", po::value<bool>(&p.control.bott_proc)->default_value(true)," ")
        ("control.bott_diff", po::value<double>(&p.control.bott_diff)->default_value(1.0e-7), " ")
        ;

    cfg.add_options()
        ("mesh.mesh_file", po::value<std::string>(&p.mesh.mesh_file)->default_value("default")," ")
        ("mesh.rs_levels", po::value<int>(&p.mesh.rs_levels)->default_value(2)," ")
        ("mesh.rp_levels", po::value<int>(&p.mesh.rp_levels)->default_value(0)," ")
        ("mesh.partition_type", po::value<int>(&p.mesh.partition_type)->default_value(0),
         "Partition type as an integer")
        ("mesh.order_v", po::value<int>(&p.mesh.order_v)->default_value(2),"Order (degree) of the kinematic finite element space.")
        ("mesh.order_e", po::value<int>(&p.mesh.order_e)->default_value(1),"Order (degree) of the thermodynamic finite element space.")
        ("mesh.order_q", po::value<int>(&p.mesh.order_q)->default_value(-1),"Order  of the integration rule.")
        ("mesh.local_refinement", po::value<bool>(&p.mesh.local_refinement)->default_value(false), " ")
        ("mesh.l2_basis", po::value<>(&p.mesh.l2_basis)->default_value(1)," ")
        ;

    cfg.add_options()
        ("mat.plastic", po::value<bool>(&p.mat.plastic)->default_value(true), " ")
        ("mat.viscoplastic", po::value<bool>(&p.mat.viscoplastic)->default_value(false), " ")
        ("mat.rho", po::value<std::string>(&p.mat.rho)->default_value("[2700.0]"),"Material indicators '[d0, d1, d2, ...]")
        ("mat.lambda", po::value<std::string>(&p.mat.lambda)->default_value("[3e10]"),"Material indicators '[d0, d1, d2, ...]")
        ("mat.mu", po::value<std::string>(&p.mat.mu)->default_value("[3e10]"),"Material indicators '[d0, d1, d2, ...]")
        ("mat.weak_rad", po::value<double>(&p.mat.weak_rad)->default_value(1.0e3), "circular weakzone")//
        ("mat.weak_x", po::value<double>(&p.mat.weak_x)->default_value(50.0e3), " x coord of circular")//
        ("mat.weak_y", po::value<double>(&p.mat.weak_y)->default_value(2.0e3), "y coord of circular")  //
        ("mat.weak_z", po::value<double>(&p.mat.weak_z)->default_value(0.0), "z coord of circular")    //
        ("mat.ini_pls", po::value<double>(&p.mat.ini_pls)->default_value(0.5), "initial plasticity")   //
        ("mat.tension_cutoff", po::value<std::string>(&p.mat.tension_cutoff)->default_value("[0.0]"),"Material indicators '[d0, d1, d2, ...]")
        ("mat.cohesion0", po::value<std::string>(&p.mat.cohesion0)->default_value("[44.0e6]"),"Material indicators '[d0, d1, d2, ...]")
        ("mat.cohesion1", po::value<std::string>(&p.mat.cohesion1)->default_value("[44.0e6]"),"Material indicators '[d0, d1, d2, ...]")
        ("mat.friction_angle0", po::value<std::string>(&p.mat.friction_angle0)->default_value("[30.0]"),"Material indicators '[d0, d1, d2, ...]")
        ("mat.friction_angle1", po::value<std::string>(&p.mat.friction_angle1)->default_value("[30.0]"),"Material indicators '[d0, d1, d2, ...]")
        ("mat.dilation_angle0", po::value<std::string>(&p.mat.dilation_angle0)->default_value("[0.0]"),"Material indicators '[d0, d1, d2, ...]")
        ("mat.dilation_angle1", po::value<std::string>(&p.mat.dilation_angle1)->default_value("[0.0]"),"Material indicators '[d0, d1, d2, ...]")
        ("mat.pls0", po::value<std::string>(&p.mat.pls0)->default_value("[0.0]"),"Material indicators '[d0, d1, d2, ...]")
        ("mat.pls1", po::value<std::string>(&p.mat.pls1)->default_value("[0.5]"),"Material indicators '[d0, d1, d2, ...]")
        ("mat.plastic_viscosity", po::value<std::string>(&p.mat.plastic_viscosity)->default_value("[1.0]"),"Material indicators '[d0, d1, d2, ...]")
        ;
    cfg.add_options()
        ("tmop.tmop", po::value<bool>(&p.tmop.tmop)->default_value(false), " ")
        ("tmop.amr", po::value<bool>(&p.tmop.amr)->default_value(false), " ")
        ("tmop.ale", po::value<double>(&p.tmop.ale)->default_value(0.5), " ")
        ("tmop.remesh_steps", po::value<int>(&p.tmop.remesh_steps)->default_value(50000), " ")
        ("tmop.mesh_poly_deg", po::value<int>(&p.tmop.mesh_poly_deg)->default_value(2), " ")
        ("tmop.jitter", po::value<double>(&p.tmop.jitter)->default_value(0.0), " ")
        ("tmop.metric_id", po::value<int>(&p.tmop.metric_id)->default_value(2), " ")
        ("tmop.target_id", po::value<int>(&p.tmop.target_id)->default_value(1), " ")
        ("tmop.lim_const", po::value<double>(&p.tmop.lim_const)->default_value(0.0), " ")
        ("tmop.adapt_lim_const", po::value<double>(&p.tmop.adapt_lim_const)->default_value(0.0), " ")
        ("tmop.quad_type", po::value<int>(&p.tmop.quad_type)->default_value(1), " ")
        ("tmop.quad_order", po::value<int>(&p.tmop.quad_order)->default_value(8), " ")
        ("tmop.solver_type", po::value<int>(&p.tmop.solver_type)->default_value(0), " ")
        ("tmop.solver_iter", po::value<int>(&p.tmop.solver_iter)->default_value(20), " ")
        ("tmop.solver_rtol", po::value<double>(&p.tmop.solver_rtol)->default_value(1e-10), " ")
        ("tmop.solver_art_type", po::value<int>(&p.tmop.solver_art_type)->default_value(0), " ")
        ("tmop.lin_solver", po::value<int>(&p.tmop.lin_solver)->default_value(2), " ")
        ("tmop.max_lin_iter", po::value<int>(&p.tmop.max_lin_iter)->default_value(100), " ")
        ("tmop.move_bnd", po::value<bool>(&p.tmop.move_bnd)->default_value(false), " ")
        ("tmop.combomet", po::value<int>(&p.tmop.combomet)->default_value(0), " ")
        ("tmop.bal_expl_combo", po::value<bool>(&p.tmop.bal_expl_combo)->default_value(false), " ")
        ("tmop.hradaptivity", po::value<bool>(&p.tmop.hradaptivity)->default_value(false), " ")
        ("tmop.h_metric_id", po::value<int>(&p.tmop.h_metric_id)->default_value(-1), " ")
        ("tmop.normalization", po::value<bool>(&p.tmop.normalization)->default_value(false), " ")
        ("tmop.verbosity_level", po::value<int>(&p.tmop.verbosity_level)->default_value(0), " ")
        ("tmop.fdscheme", po::value<bool>(&p.tmop.fdscheme)->default_value(false), " ")
        ("tmop.adapt_eval", po::value<int>(&p.tmop.adapt_eval)->default_value(0), " ")
        ("tmop.exactaction", po::value<bool>(&p.tmop.exactaction)->default_value(false), " ")
        ("tmop.n_hr_iter", po::value<int>(&p.tmop.n_hr_iter)->default_value(5), " ")
        ("tmop.n_h_iter", po::value<int>(&p.tmop.n_h_iter)->default_value(1), " ")
        ("tmop.mesh_node_ordering", po::value<int>(&p.tmop.mesh_node_ordering)->default_value(0), " ")
        ("tmop.barrier_type", po::value<int>(&p.tmop. barrier_type)->default_value(0), " ")
        ("tmop.worst_case_type", po::value<int>(&p.tmop.worst_case_type)->default_value(0), " ")
        ("tmop.tmop_cond_num", po::value<double>(&p.tmop.tmop_cond_num)->default_value(0.5), " ")
        ;
}


static void read_parameters_from_file
(const char* filename,
 const po::options_description cfg,
 po::variables_map &vm)
{
    try {
        po::store(po::parse_config_file<char>(filename, cfg), vm);
        po::notify(vm);
    }
    catch (const boost::program_options::multiple_occurrences& e) {
        std::cerr << e.what() << " from option: " << e.get_option_name() << '\n';
        std::exit(1);
    }
    catch (std::exception& e) {
        std::cerr << "Error reading config_file '" << filename << "'\n";
        std::cerr << e.what() << "\n";
        std::exit(1);
    }
}

void get_input_parameters(const char* filename, Param& p)
{
    po::options_description cfg("Config file options");
    po::variables_map vm;

    declare_parameters(cfg, p);
    // print help message
    if (std::strncmp(filename, "-h", 3) == 0 ||
        std::strncmp(filename, "--help", 7) == 0) {
        std::cout << cfg;
        std::exit(0);
    }
    read_parameters_from_file(filename, cfg, vm);
    // validate_parameters(vm, p);
}



#if 0
template<class T>
static int read_numbers(const std::string &input, std::vector<T> &vec, int len)
{
    /* Read 'len' numbers from input.
     * The format of input must be '[n0, n1, n2]' or '[n0, n1, n2,]' (with a trailing ,),
     * for len=3.
     */

    std::istringstream stream(input);
    vec.resize(len);

    char sentinel;

    stream >> sentinel;
    if (sentinel != '[') return 1;

    for (int i=0; i<len; ++i) {
        stream >> vec[i];

        if (i == len-1) break;

        // consume ','
        char sep;
        stream >> sep;
        if (sep != ',') return 1;
    }

    stream >> sentinel;
    if (sentinel == ',') stream >> sentinel;
    if (sentinel != ']') return 1;

    if (! stream.good()) return 1;

    // success
    return 0;
}

template<class T>
static void get_numbers(const po::variables_map &vm, const char *name,
                        std::vector<T> &values, int len, int optional_size=0)
{
    if ( ! vm.count(name) ) {
        std::cerr << "Error: " << name << " is not provided.\n";
        std::exit(1);
    }

    std::string str = vm[name].as<std::string>();
    int err = read_numbers(str, values, len);
    if (err && optional_size) {
        err = read_numbers(str, values, optional_size);
    }

    if (err) {
        std::cerr << "Error: incorrect format for " << name << ",\n"
                  << "       must be '[d0, d1, d2, ...]'\n";
        std::exit(1);
    }
}

static void validate_parameters(const po::variables_map &vm, Param &p)
{
    std::cout << "Checking consistency of input parameters...\n";

    //
    // stopping condition and output interval are based on either model time or step
    //
    if ( ! (vm.count("sim.max_steps") || vm.count("sim.max_time_in_yr")) ) {
        std::cerr << "Must provide either sim.max_steps or sim.max_time_in_yr\n";
        std::exit(1);
    }
    if ( ! vm.count("sim.max_steps") )
        p.sim.max_steps = std::numeric_limits<int>::max();
    if ( ! vm.count("sim.max_time_in_yr") )
        p.sim.max_time_in_yr = std::numeric_limits<double>::max();

    if ( ! (vm.count("sim.output_step_interval") || vm.count("sim.output_time_interval_in_yr")) ) {
        std::cerr << "Must provide either sim.output_step_interval or sim.output_time_interval_in_yr\n";
        std::exit(1);
    }
    if ( ! vm.count("sim.output_step_interval") )
        p.sim.output_step_interval = std::numeric_limits<int>::max();
    if ( ! vm.count("sim.output_time_interval_in_yr") )
        p.sim.output_time_interval_in_yr = std::numeric_limits<double>::max();

    //
    // These parameters are required when restarting
    //
    if (p.sim.is_restarting) {
        if ( ! vm.count("sim.restarting_from_modelname") ) {
            std::cerr << "Must provide sim.restarting_from_modelname when restarting.\n";
            std::exit(1);
        }
        if ( ! vm.count("sim.restarting_from_frame") ) {
            std::cerr << "Must provide sim.restarting_from_frame when restarting.\n";
            std::exit(1);
        }
    }

    //
    // these parameters are required in mesh.meshing_option == 2
    //
    if (p.mesh.meshing_option == 2) {
        if ( ! vm.count("mesh.refined_zonex") ||
#ifdef THREED
             ! vm.count("mesh.refined_zoney") ||
#endif
             ! vm.count("mesh.refined_zonez") ) {
        std::cerr << "Must provide mesh.refined_zonex, "
#ifdef THREED
                  << "mesh.refined_zoney, "
#endif
                  << "mesh.refined_zonez.\n";
        std::exit(1);
        }

        /* get 2 numbers from the string */
        double_vec tmp;
        int err;
        std::string str;
        str = vm["mesh.refined_zonex"].as<std::string>();
        err = read_numbers(str, tmp, 2);
        if (err || tmp[0] < 0 || tmp[1] > 1 || tmp[0] > tmp[1]) {
            std::cerr << "Error: incorrect value for mesh.refine_zonex,\n"
                      << "       must in this format '[d0, d1]', 0 <= d0 <= d1 <= 1.\n";
            std::exit(1);
        }
        p.mesh.refined_zonex.first = tmp[0];
        p.mesh.refined_zonex.second = tmp[1];
#ifdef THREED
        str = vm["mesh.refined_zoney"].as<std::string>();
        err = read_numbers(str, tmp, 2);
        if (err || tmp[0] < 0 || tmp[1] > 1 || tmp[0] > tmp[1]) {
            std::cerr << "Error: incorrect value for mesh.refine_zoney,\n"
                      << "       must in this format '[d0, d1]', 0 <= d0 <= d1 <= 1.\n";
            std::exit(1);
        }
        p.mesh.refined_zoney.first = tmp[0];
        p.mesh.refined_zoney.second = tmp[1];
#endif
        str = vm["mesh.refined_zonez"].as<std::string>();
        err = read_numbers(str, tmp, 2);
        if (err || tmp[0] < 0 || tmp[1] > 1 || tmp[0] > tmp[1]) {
            std::cerr << "Error: incorrect value for mesh.refine_zonez,\n"
                      << "       must in this format '[d0, d1]', 0 <= d0 <= d1 <= 1.\n";
            std::exit(1);
        }
        p.mesh.refined_zonez.first = tmp[0];
        p.mesh.refined_zonez.second = tmp[1];
    }

    if (p.mesh.smallest_size > p.mesh.largest_size) {
        std::cerr << "Error: mesh.smallest_size is greater than mesh.largest_size.\n";
        std::exit(1);
    }

#ifdef THREED
    if (p.mesh.remeshing_option == 2) {
        std::cerr << "Error: mesh.remeshing_option=2 is not available in 3D.\n";
        std::exit(1);
    }
#endif

    //
    // bc
    //
    {
        if ( p.bc.has_winkler_foundation && p.control.gravity == 0 ) {
            p.bc.has_winkler_foundation = 0;
            std::cerr << "Warning: no gravity, Winkler foundation is turned off.\n";
        }
        if ( p.bc.has_winkler_foundation && p.bc.vbc_z0 != 0 ) {
            p.bc.vbc_z0 = 0;
            std::cerr << "Warning: Winkler foundation is turned on, setting bc.vbc_z0 to 0.\n";
        }
        if ( p.bc.has_water_loading && p.control.gravity == 0 ) {
            p.bc.has_water_loading = 0;
            std::cerr << "Warning: no gravity, water loading is turned off.\n";
        }
        if ( p.bc.has_water_loading && p.bc.vbc_z1 != 0 ) {
            p.bc.vbc_z1 = 0;
            std::cerr << "Warning: water loading is turned on, setting bc.vbc_z1 to 0.\n";
        }

        if ( p.bc.vbc_z0 > 3) {
            std::cerr << "Error: bc.vbc_z0 is not 0, 1, 2, or 3.\n";
            std::exit(1);
        }
        if ( p.bc.vbc_z1 > 3) {
            std::cerr << "Error: bc.vbc_z0 is not 0, 1, 2, or 3.\n";
            std::exit(1);
        }
        if ( p.bc.vbc_n0 != 1 && p.bc.vbc_n0 != 3 && p.bc.vbc_n0 != 11 && p.bc.vbc_n0 != 13 ) {
            std::cerr << "Error: bc.vbc_n0 is not 1, 3, 11, or 13.\n";
            std::exit(1);
        }
        if ( p.bc.vbc_n1 != 1 && p.bc.vbc_n1 != 3 && p.bc.vbc_n1 != 11 && p.bc.vbc_n1 != 13 ) {
            std::cerr << "Error: bc.vbc_n1 is not 1, 3, 11, or 13.\n";
            std::exit(1);
        }
        if ( p.bc.vbc_n2 != 1 && p.bc.vbc_n2 != 3 && p.bc.vbc_n2 != 11 && p.bc.vbc_n2 != 13 ) {
            std::cerr << "Error: bc.vbc_n2 is not 1, 3, 11, or 13.\n";
            std::exit(1);
        }
        if ( p.bc.vbc_n3 != 1 && p.bc.vbc_n3 != 3 && p.bc.vbc_n3 != 11 && p.bc.vbc_n3 != 13 ) {
            std::cerr << "Error: bc.vbc_n3 is not 1, 3, 11, or 13.\n";
            std::exit(1);
        }
    }

    //
    // control
    //
    {
        if ( p.control.dt_fraction < 0 || p.control.dt_fraction > 1 ) {
            std::cerr << "Error: control.dt_fraction must be between 0 and 1.\n";
            std::exit(1);
        }
        if ( p.control.damping_factor < 0 || p.control.damping_factor > 1 ) {
            std::cerr << "Error: control.damping_factor must be between 0 and 1.\n";
            std::exit(1);
        }

    }

    //
    // ic
    //
    {
        if ( p.ic.mattype_option == 1) {
            get_numbers(vm, "ic.layer_mattypes", p.ic.layer_mattypes, p.ic.num_mattype_layers);
            get_numbers(vm, "ic.mattype_layer_depths", p.ic.mattype_layer_depths, p.ic.num_mattype_layers-1);
            // mattype_layer_depths must be already sorted
            if (! std::is_sorted(p.ic.mattype_layer_depths.begin(), p.ic.mattype_layer_depths.end())) {
                std::cerr << "Error: the content of ic.mattype_layer_depths is not ordered from"
                    " small to big values.\n";
                std::exit(1);
            }
        }
    }

    //
    // marker
    //
    {

    }

    //
    // material properties
    //
    {
        std::string str = vm["mat.rheology_type"].as<std::string>();
        if (str == std::string("elastic"))
            p.mat.rheol_type = MatProps::rh_elastic;
        else if (str == std::string("viscous"))
            p.mat.rheol_type = MatProps::rh_viscous;
        else if (str == std::string("maxwell"))
            p.mat.rheol_type = MatProps::rh_maxwell;
        else if (str == std::string("elasto-plastic"))
            p.mat.rheol_type = MatProps::rh_ep;
        else if (str == std::string("elasto-visco-plastic"))
            p.mat.rheol_type = MatProps::rh_evp;
        else {
            std::cerr << "Error: unknown rheology: '" << str << "'\n";
            std::exit(1);
        }

#ifdef THREED
        if ( p.mat.is_plane_strain ) {
            p.mat.is_plane_strain = false;
            std::cerr << "Warning: mat.is_plane_strain is not avaiable in 3D.\n";
        }
#endif

        if (p.mat.phase_change_option != 0 && p.mat.nmat == 1) {
            std::cerr << "Error: mat.phase_change_option is chosen, but mat.num_materials is 1.\n";
            std::exit(1);
        }
        if (p.mat.phase_change_option == 1 && p.mat.nmat < 8) {
            std::cerr << "Error: mat.phase_change_option is 1, but mat.num_materials is less than 8.\n";
            std::exit(1);
        }

        if (p.mat.nmat < 1) {
            std::cerr << "Error: mat.num_materials must be greater than 0.\n";
            std::exit(1);
        }

        if (p.mat.nmat == 1 && p.control.ref_pressure_option != 0) {
            p.control.ref_pressure_option = 0;
            std::cerr << "Warning: mat.num_materials is 1, using simplest control.ref_pressure_option.\n";
        }
        if (p.mat.nmat == 1 && p.markers.replenishment_option != 1) {
            p.markers.replenishment_option = 1;
            std::cerr << "Warning: mat.num_materials is 1, using simplest markers.replenishment_option.\n";
        }

        get_numbers(vm, "mat.rho0", p.mat.rho0, p.mat.nmat, 1);
        get_numbers(vm, "mat.alpha", p.mat.alpha, p.mat.nmat, 1);

        get_numbers(vm, "mat.bulk_modulus", p.mat.bulk_modulus, p.mat.nmat, 1);
        get_numbers(vm, "mat.shear_modulus", p.mat.shear_modulus, p.mat.nmat, 1);

        get_numbers(vm, "mat.visc_exponent", p.mat.visc_exponent, p.mat.nmat, 1);
        get_numbers(vm, "mat.visc_coefficient", p.mat.visc_coefficient, p.mat.nmat, 1);
        get_numbers(vm, "mat.visc_activation_energy", p.mat.visc_activation_energy, p.mat.nmat, 1);

        get_numbers(vm, "mat.heat_capacity", p.mat.heat_capacity, p.mat.nmat, 1);
        get_numbers(vm, "mat.therm_cond", p.mat.therm_cond, p.mat.nmat, 1);

        get_numbers(vm, "mat.pls0", p.mat.pls0, p.mat.nmat, 1);
        get_numbers(vm, "mat.pls1", p.mat.pls1, p.mat.nmat, 1);
        get_numbers(vm, "mat.cohesion0", p.mat.cohesion0, p.mat.nmat, 1);
        get_numbers(vm, "mat.cohesion1", p.mat.cohesion1, p.mat.nmat, 1);
        get_numbers(vm, "mat.friction_angle0", p.mat.friction_angle0, p.mat.nmat, 1);
        get_numbers(vm, "mat.friction_angle1", p.mat.friction_angle1, p.mat.nmat, 1);
        get_numbers(vm, "mat.dilation_angle0", p.mat.dilation_angle0, p.mat.nmat, 1);
        get_numbers(vm, "mat.dilation_angle1", p.mat.dilation_angle1, p.mat.nmat, 1);
    }

}
#endif