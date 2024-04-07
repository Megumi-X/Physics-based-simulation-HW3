#include "deformable/include/deformable_surface_simulator.hpp"
#include "basic/include/sparse_matrix.hpp"

namespace backend {
namespace deformable {

// For now the material is hard-coded.
// A material whose energy density Psi is quadratic w.r.t. the strain tensor E.
// Psi(F) = 0.5 * E^T C E, where E = 0.5 * vec(F^T F - I).
static const Matrix4r C = Eigen::Matrix<real, 16, 1>{ 2500., 0., 0., 2000., 0., 500., 0., 0., 0., 0., 500., 0., 2000., 0., 0., 2500. }.reshaped(4, 4);
static const real ComputeEnergyDensityFromStrainTensor(const Matrix2r& E) { return 0.5 * E.reshaped().dot(C * E.reshaped()); }
// The stress tensor P is always symmetric.
static const Matrix2r ComputeStressTensorFromStrainTensor(const Matrix2r& E) { return (C * E.reshaped()).reshaped(2, 2); }

const real Simulator::ComputeStretchingAndShearingEnergy(const Matrix3Xr& position) const {
    real energy = 0;
    for (integer e = 0; e < static_cast<integer>(elements_.cols()); ++e) {
        const Eigen::Matrix<real, 3, 2> F = position(Eigen::all, elements_.col(e)) * D_inv_[e];
        const Matrix2r E = (F.transpose() * F - Matrix2r::Identity()) / 2;
        energy += ComputeEnergyDensityFromStrainTensor(E);
    }
    // The rest shape area is hard-coded to be 0.005 in this homework.
    return 0.005 * energy;
}

const Matrix3Xr Simulator::ComputeStretchingAndShearingForce(const Matrix3Xr& position) const {
    const integer element_num = static_cast<integer>(elements_.cols());
    std::vector<Matrix3r> gradients(element_num);
    for (integer e = 0; e < element_num; ++e) {
        const Eigen::Matrix<real, 3, 2> F = position(Eigen::all, elements_.col(e)) * D_inv_[e];
        const Matrix2r E = (F.transpose() * F - Matrix2r::Identity()) / 2;
        const auto P = ComputeStressTensorFromStrainTensor(E);
        // Derive the gradient dE/dx, where x stands for a 3x3 matrix position(Eigen::all, elements_.col(e)).
        // TODO.
        gradients[e] = Matrix3r::Zero();
        gradients[e] = 0.005 * F * P * D_inv_[e].transpose();
        /////////////////////////////////
    }

    Matrix3Xr gradient = Matrix3Xr::Zero(3, position.cols());
    for (integer k = 0; k < static_cast<integer>(position.cols()); ++k) {
        for (const auto& tuple : stretching_and_shearing_gradient_map_[k]) {
            const integer e = tuple[0];
            const integer i = tuple[1];
            gradient.col(k) += gradients[e].col(i);
        }
    }

    // Force is negative gradient.
    return -gradient;
}

const SparseMatrixXr Simulator::ComputeStretchingAndShearingHessian(const Matrix3Xr& position) const {
    const integer element_num = static_cast<integer>(elements_.cols());
    std::vector<Matrix9r> hess_nonzeros;
    hess_nonzeros.reserve(element_num);
    for (integer e = 0; e < element_num; ++e) {
        const Eigen::Matrix<real, 3, 2> F = position(Eigen::all, elements_.col(e)) * D_inv_[e];
        const Matrix2r E = (F.transpose() * F - Matrix2r::Identity()) / 2;
        const auto P = ComputeStressTensorFromStrainTensor(E);
        // Derive the Hessian d^2E/dx^2, where x stands for a column vector concatenated by the vertices x1, x2, x3.
        // (You do not need to consider the SPD projection issue.)
        // TODO.
        Matrix2r dedx[3][3];
        for (integer d = 0; d < 3; d++) {
            for (integer i = 0; i < 3; i++) {
                dedx[d][i] = (D_inv_[e].row(i).transpose() * F.row(d) + F.row(d).transpose() * D_inv_[e].row(i)) / 2;
            }
        }
        hess_nonzeros.push_back(Matrix9r::Zero());
        for (integer d = 0; d < 3; d++) {
            for (integer c = 0; c < 3; c++) {
                for (integer i = 0; i < 3; i++) {
                    for (integer j = 0; j < 3; j++) {
                        const integer idx = i * 3 + d;
                        const integer jdx = j * 3 + c;
                        integer part_1 = (c == d) * D_inv_[e].row(j) * P * D_inv_[e].row(i).transpose();
                        integer part_2 = dedx[d][i].reshaped().dot(C * dedx[c][j].reshaped());
                        hess_nonzeros[e](idx, jdx) = 0.005 * (part_1 + part_2);
                    }
                }
            }
        }
        /////////////////////////////////
    }

    SparseMatrixXr ret(stretching_and_shearing_hessian_);
    for (integer k = 0; k < stretching_and_shearing_hessian_nonzero_num_; ++k) {
        real val = 0;
        for (const auto& arr : stretching_and_shearing_hessian_nonzero_map_[k]) {
            const integer e = arr[0];
            const integer i = arr[1];
            const integer j = arr[2];
            val += hess_nonzeros[e](i, j);
        }
        ret.valuePtr()[k] = val;
    }

    return ret;
}

}
}