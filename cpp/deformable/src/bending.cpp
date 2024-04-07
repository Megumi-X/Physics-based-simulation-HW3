#include "deformable/include/deformable_surface_simulator.hpp"
#include "basic/include/sparse_matrix.hpp"

namespace backend {
namespace deformable {

static const real ComputeDihedralAngleFromNonUnitNormal(const Vector3r& normal, const Vector3r& other_normal) {
    const real sin_angle = normal.cross(other_normal).norm();
    const real cos_angle = normal.dot(other_normal);
    const real angle = std::atan2(sin_angle, cos_angle);
    return angle;
}

static const Vector3r ComputeNormal(const Matrix3r& vertices) {
    // This is the normal direction vector that is not normalized.
    // You may assume that in this homework the area of a triangle does not shrink below 1e-5,
    // therefore the normal direction (a x b)/||a x b|| does not suffer from numerical issues.
    return (vertices.col(1) - vertices.col(0)).cross(vertices.col(2) - vertices.col(1));
}

// In this homework, the rest shape area is fixed and hard-coded to be 0.005. In general it could be computed from the inputs. 
const real Simulator::ComputeBendingEnergy(const Matrix3Xr& position) const {
    // Loop over all edges.
    const integer element_num = static_cast<integer>(elements_.cols());
    real energy = 0;
    for (integer e = 0; e < element_num; ++e) {
        // Compute normal.
        const Vector3r normal = ComputeNormal(position(Eigen::all, elements_.col(e)));
        for (integer i = 0; i < 3; ++i) {
            const TriangleEdgeInfo& info = triangle_edge_info_[e][i];
            // We only care about internal edges and only computes each edge once.
            if (info.other_triangle == -1 || e > info.other_triangle) continue;
            const Vector3r other_normal = ComputeNormal(position(Eigen::all, elements_.col(info.other_triangle)));
            const real angle = ComputeDihedralAngleFromNonUnitNormal(normal, other_normal);
            const real rest_shape_edge_length = info.edge_length;
            const real diamond_area = 0.01 / 3;
            // TODO.

            /////////////////////////////////////////////
        }
    }

    return bending_stiffness_ * energy;
}

const Matrix3Xr Simulator::ComputeBendingForce(const Matrix3Xr& position) const {
    // TODO.
    return Matrix3Xr::Zero(3, position.cols());
    /////////////////////////////////////
}

const SparseMatrixXr Simulator::ComputeBendingHessian(const Matrix3Xr& position) const {
    // TODO.
    const integer hess_size = static_cast<integer>(position.cols()) * 3;
    return FromTriplet(hess_size, hess_size, {});
    /////////////////////////////////////
}

}
}