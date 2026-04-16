#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <random>
#include <cmath>

namespace py = pybind11;

void jitted_physics_core_cpp(
    py::array_t<double> x_act_arr, py::array_t<double> y_act_arr, py::array_t<double> z_act_arr,
    py::array_t<double> ux_arr, py::array_t<double> uy_arr, py::array_t<double> uz_arr,
    py::array_t<double> tau_act_arr, py::array_t<double> D_act_arr, py::array_t<double> Z_act_arr,
    double v_th_x, double v_th_y, double v_th_z,
    double Ex, double Ey, double Ez,
    double dt, double gravity,
    double xmin, double xmax, double ymax, double L1, double theta,
    py::array_t<double> x_new_arr, py::array_t<double> y_new_arr, py::array_t<double> z_new_arr,
    py::array_t<bool> hit_wall_arr, py::array_t<bool> hit_bottom_arr
) {
    // Acquire bounds checking bypassing buffers
    auto x_act = x_act_arr.unchecked<1>();
    auto y_act = y_act_arr.unchecked<1>();
    auto z_act = z_act_arr.unchecked<1>();
    auto ux = ux_arr.unchecked<1>();
    auto uy = uy_arr.unchecked<1>();
    auto uz = uz_arr.unchecked<1>();
    auto tau_act = tau_act_arr.unchecked<1>();
    auto D_act = D_act_arr.unchecked<1>();
    auto Z_act = Z_act_arr.unchecked<1>();
    
    // Acquire mutable output arrays
    auto x_new = x_new_arr.mutable_unchecked<1>();
    auto y_new = y_new_arr.mutable_unchecked<1>();
    auto z_new = z_new_arr.mutable_unchecked<1>();
    auto hit_wall = hit_wall_arr.mutable_unchecked<1>();
    auto hit_bottom = hit_bottom_arr.mutable_unchecked<1>();

    int n_active = x_act.shape(0);
    double R_main = ymax;
    double R_branch = R_main / std::sqrt(2.0);
    double cos_theta = std::cos(theta);
    double tan_theta = std::tan(theta);
    double R_main_sq = R_main * R_main;
    double R_branch_sq = R_branch * R_branch;

    // Standard high-performance C++ MT19937 RNG framework 
    // initialized safely preventing seed-duplication bugs on threads
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> d(0.0, 1.0);

    for (int i = 0; i < n_active; ++i) {
        // Drift Physics
        double v_settling_y = -tau_act(i) * gravity;
        double total_vx = ux(i) + v_th_x + Z_act(i) * Ex;
        double total_vy = uy(i) + v_th_y + Z_act(i) * Ey + v_settling_y;
        double total_vz = uz(i) + v_th_z + Z_act(i) * Ez;

        // Diffusion (Stochastic Euler Bounds)
        double sigma = std::sqrt(2.0 * D_act(i) * dt);
        double dW_x = d(gen) * sigma;
        double dW_y = d(gen) * sigma;
        double dW_z = d(gen) * sigma;

        double nx = x_act(i) + total_vx * dt + dW_x;
        double ny = y_act(i) + total_vy * dt + dW_y;
        double nz = z_act(i) + total_vz * dt + dW_z;

        // Boundary Detectors
        bool hw = false;
        bool hb = false;

        if (nx <= L1) {
            if (ny * ny + nz * nz >= R_main_sq) {
                hw = true;
            }
        } else {
            double xb = nx - L1;
            double yc_up = xb * tan_theta;
            double yc_down = -xb * tan_theta;

            double dist_up2 = (ny - yc_up) * (ny - yc_up) * cos_theta * cos_theta + nz * nz;
            double dist_down2 = (ny - yc_down) * (ny - yc_down) * cos_theta * cos_theta + nz * nz;

            if (dist_up2 >= R_branch_sq && dist_down2 >= R_branch_sq) {
                hw = true;
            }
        }

        if (nx >= xmax) {
            hb = true;
        }

        x_new(i) = nx;
        y_new(i) = ny;
        z_new(i) = nz;
        hit_wall(i) = hw;
        hit_bottom(i) = hb;
    }
}

// Map the signature internally via the module extension PyBind11 hook
PYBIND11_MODULE(psat_cpp_core, m) {
    m.doc() = "Native C++ implementation of the PSAT Euler-Maruyama integration loop.";
    m.def("jitted_physics_core_cpp", &jitted_physics_core_cpp,
          "Advances active geometry boundary markers via parallel arrays natively on the C stack.");
}
