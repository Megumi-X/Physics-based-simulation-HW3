#include "deformable/include/deformable_surface_simulator.hpp"
#include "basic/include/sparse_matrix.hpp"

namespace backend {
namespace deformable {

    const real Simulator::ComputeContactEnergy(const Matrix3Xr& position) const {
        real energy = 0;
        for (integer i = 0; i < static_cast<integer>(position.cols()); ++i) {
            const Vector3r p = position.col(i);
            const real distance = p.norm() - 1;
            if (distance > contact_delta_) continue;
            energy += contact_stiffness_ * (distance - contact_delta_) * (distance - contact_delta_) / 2;
        }
        return energy;
    }

    const Matrix3Xr Simulator::ComputeContactForce(const Matrix3Xr& position) const {
        Matrix3Xr force = Matrix3Xr::Zero(3, position.cols());
        for (integer i = 0; i < static_cast<integer>(position.cols()); ++i) {
            const Vector3r p = position.col(i);
            const real distance = p.norm() - 1;
            if (distance > contact_delta_) continue;
            force.col(i) += contact_stiffness_ * (distance - contact_delta_) * p / p.norm();
        }
        return -force;
    }

    const SparseMatrixXr Simulator::ComputeContactHessian(const Matrix3Xr& position) const {
        SparseMatrixXr hessian(3 * position.cols(), 3 * position.cols());
        for (integer i = 0; i < static_cast<integer>(position.cols()); ++i) {
            const Vector3r p = position.col(i);
            const real distance = p.norm() - 1;
            if (distance > contact_delta_) continue;
            const real R = 1 + contact_delta_;
            for (integer j = 0; j < 3; ++j) {
                hessian.coeffRef(3 * i + j, 3 * i + j) += (1 - (p[(j + 1) % 3] * p[(j + 1) % 3] + p[(j + 2) % 3] * p[(j + 2) % 3]) * R / (p.norm() * p.norm() * p.norm())) * contact_stiffness_;
                for (integer k = 1; k < 3; ++k) {
                    const integer index = (j + k) % 3;
                    hessian.coeffRef(3 * i + j, 3 * i + index) += (p[j] * p[index] * R / (p.norm() * p.norm() * p.norm())) * contact_stiffness_;
                }
            }
        }
        return hessian;
    }

}
}