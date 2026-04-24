#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <omp.h>
#include <random>
#include <cmath>
#include <algorithm>

namespace py = pybind11;

void jitted_physics_core_cpp(
    py::array_t<double> x_act_arr, py::array_t<double> y_act_arr, py::array_t<double> z_act_arr,
    py::array_t<double> ux_arr, py::array_t<double> uy_arr, py::array_t<double> uz_arr,
    py::array_t<double> tau_act_arr, py::array_t<double> D_brownian_arr, py::array_t<double> Z_act_arr,
    py::array_t<int> branch_ids_arr,
    py::array_t<double> tree_starts_arr, py::array_t<double> tree_dirs_arr, 
    py::array_t<double> tree_lens_arr, py::array_t<double> tree_radii_arr,
    py::array_t<int> child_left_arr, py::array_t<int> child_right_arr,
    double v_th_x, double v_th_y, double v_th_z,
    double Ex, double Ey, double Ez,
    double dt, double gravity,
    double xmin, double xmax, double ymax,
    double turb_alpha, double mu, double rho_f,
    py::array_t<double> x_new_arr, py::array_t<double> y_new_arr, py::array_t<double> z_new_arr,
    py::array_t<bool> hit_wall_arr, py::array_t<bool> hit_bottom_arr
) {
    auto x_act = x_act_arr.unchecked<1>();
    auto y_act = y_act_arr.unchecked<1>();
    auto z_act = z_act_arr.unchecked<1>();
    auto ux = ux_arr.unchecked<1>();
    auto uy = uy_arr.unchecked<1>();
    auto uz = uz_arr.unchecked<1>();
    auto tau_act = tau_act_arr.unchecked<1>();
    auto D_brownian = D_brownian_arr.unchecked<1>();
    auto Z_act = Z_act_arr.unchecked<1>();
    auto branch_ids = branch_ids_arr.mutable_unchecked<1>();

    auto tree_starts = tree_starts_arr.unchecked<2>();
    auto tree_dirs = tree_dirs_arr.unchecked<2>();
    auto tree_lens = tree_lens_arr.unchecked<1>();
    auto tree_radii = tree_radii_arr.unchecked<1>();
    auto child_left = child_left_arr.unchecked<1>();
    auto child_right = child_right_arr.unchecked<1>();
    
    auto x_new = x_new_arr.mutable_unchecked<1>();
    auto y_new = y_new_arr.mutable_unchecked<1>();
    auto z_new = z_new_arr.mutable_unchecked<1>();
    auto hit_wall = hit_wall_arr.mutable_unchecked<1>();
    auto hit_bottom = hit_bottom_arr.mutable_unchecked<1>();

    int n_active = x_act.shape(0);

    #pragma omp parallel
    {
        std::random_device rd;
        std::mt19937 gen(rd() ^ omp_get_thread_num()); 
        std::normal_distribution<double> dist_gen(0.0, 1.0);

        #pragma omp for
        for (int i = 0; i < n_active; ++i) {
            int b_idx = branch_ids(i);
            double b_start[3] = {tree_starts(b_idx, 0), tree_starts(b_idx, 1), tree_starts(b_idx, 2)};
            double b_dir[3] = {tree_dirs(b_idx, 0), tree_dirs(b_idx, 1), tree_dirs(b_idx, 2)};
            double b_len = tree_lens(b_idx);
            double b_rad = tree_radii(b_idx);

            // ── Turbulence Model ──────────────────────────────────────────────
            double v_mag = std::sqrt(ux(i)*ux(i) + uy(i)*uy(i) + uz(i)*uz(i));
            double Re = (rho_f * v_mag * (2.0 * b_rad)) / mu;

            double dist_from_bif = (x_act(i) - b_start[0]) * b_dir[0] + 
                                   (y_act(i) - b_start[1]) * b_dir[1] + 
                                   (z_act(i) - b_start[2]) * b_dir[2];

            double decay_len = 5.0 * b_rad;
            double D_eddy = turb_alpha * Re * std::exp(-std::max(0.0, dist_from_bif) / decay_len);
            double D_total = D_brownian(i) + D_eddy;

            // ── Deterministic Drift ───────────────────────────────────────────
            double v_settling_y = -tau_act(i) * gravity;
            double total_vx = ux(i) + v_th_x + Z_act(i) * Ex;
            double total_vy = uy(i) + v_th_y + Z_act(i) * Ey + v_settling_y;
            double total_vz = uz(i) + v_th_z + Z_act(i) * Ez;

            // ── Stochastic Diffusion ──────────────────────────────────────────
            double sigma = std::sqrt(2.0 * D_total * dt);
            double dW_x = dist_gen(gen) * sigma;
            double dW_y = dist_gen(gen) * sigma;
            double dW_z = dist_gen(gen) * sigma;

            double nx = x_act(i) + total_vx * dt + dW_x;
            double ny = y_act(i) + total_vy * dt + dW_y;
            double nz = z_act(i) + total_vz * dt + dW_z;

            // ── Boundary Detectors ────────────────────────────────────────────
            bool hw = false;
            bool hb = false;

            double vp_x = nx - b_start[0];
            double vp_y = ny - b_start[1];
            double vp_z = nz - b_start[2];

            double proj = vp_x * b_dir[0] + vp_y * b_dir[1] + vp_z * b_dir[2];
            double dist_sq = (vp_x*vp_x + vp_y*vp_y + vp_z*vp_z) - proj*proj;

            if (dist_sq >= b_rad * b_rad) {
                hw = true;
            }

            if (proj >= b_len) {
                int left = child_left(b_idx);
                int right = child_right(b_idx);
                if (left == -1) {
                    hb = true;
                } else {
                    if (ny >= (b_start[1] + b_dir[1] * b_len)) {
                        branch_ids(i) = left;
                    } else {
                        branch_ids(i) = right;
                    }
                }
            }

            x_new(i) = nx;
            y_new(i) = ny;
            z_new(i) = nz;
            hit_wall(i) = hw;
            hit_bottom(i) = hb;
        }
    }
}

PYBIND11_MODULE(psat_cpp_core, m) {
    m.doc() = "Native C++ implementation of the PSAT Euler-Maruyama integration loop.";
    m.def("jitted_physics_core_cpp", &jitted_physics_core_cpp,
          "Advances active geometry boundary markers via parallel arrays natively on the C stack.");
}
