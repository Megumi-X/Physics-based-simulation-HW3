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

static const Matrix3r CrossProductMatrix(const Vector3r& v) {
    Matrix3r ret;
    ret << 0, -v(2), v(1),
           v(2), 0, -v(0),
           -v(1), v(0), 0;
    return ret;
}

static const Matrix3Xr ComputeNormalGradient(const Matrix3r& vertices) {
    Matrix3Xr ret = Matrix3Xr::Zero(3, 9);
    ret.block(0, 0, 3, 3) = CrossProductMatrix(vertices.col(2) - vertices.col(1));
    ret.block(0, 3, 3, 3) = CrossProductMatrix(vertices.col(0) - vertices.col(2));
    ret.block(0, 6, 3, 3) = CrossProductMatrix(vertices.col(1) - vertices.col(0));
    return ret;
}

static const std::pair<Vector3r, Vector3r> ComputeDihedralAngleGradient(const Vector3r& normal, const Vector3r& other_normal) {
    const real sin_angle = normal.cross(other_normal).norm();
    const real cos_angle = normal.dot(other_normal);
    const real angle = std::atan2(sin_angle, cos_angle);
    const real d_angle_d_sin = cos_angle / (cos_angle * cos_angle + sin_angle * sin_angle);
    const real d_angle_d_cos = -sin_angle / (cos_angle * cos_angle + sin_angle * sin_angle);
    Vector3r axis_normal = Vector3r::Zero();
    if (sin_angle > 1e-8 * normal.norm() * other_normal.norm()) axis_normal = normal.cross(other_normal) / sin_angle;
    const Vector3r d_sin_d_n1 = other_normal.cross(axis_normal);
    const Vector3r d_sin_d_n2 = -normal.cross(axis_normal);
    const Vector3r d_cos_d_n1 = other_normal;
    const Vector3r d_cos_d_n2 = normal;
    return { d_angle_d_sin * d_sin_d_n1 + d_angle_d_cos * d_cos_d_n1, d_angle_d_sin * d_sin_d_n2 + d_angle_d_cos * d_cos_d_n2 };
}

static const std::array<Matrix3Xr, 9> ComputeNormalHessian(const Matrix3r& vertices) {
    std::array<Matrix3Xr, 9> hess;
    hess.fill(Matrix3Xr::Zero(3, 9));
    for (integer i = 0; i < 3; i++) {
        for (integer j = 0; j < 3; j++) {
            hess[((i + 1) % 3) * 3 + j].block(0, 3 * i, 3, 3) -= CrossProductMatrix(Vector3r::Unit(j));
            hess[((i + 2) % 3) * 3 + j].block(0, 3 * i, 3, 3) += CrossProductMatrix(Vector3r::Unit(j));
        }
    }
    return hess;
}

static const std::array<Matrix3r, 4> ComputeDihedralAngleHessian(const Vector3r& normal, const Vector3r& other_normal) {
    std::array<Matrix3r, 4> hess;

    const real sin_angle = normal.cross(other_normal).norm();
    const real cos_angle = normal.dot(other_normal);
    // const real angle = std::atan2(sin_angle, cos_angle);
    const real scale = cos_angle * cos_angle + sin_angle * sin_angle;
    const real d_angle_d_sin = cos_angle / scale;
    const real d_angle_d_cos = -sin_angle / scale;
    Vector3r axis_normal = Vector3r::Zero();
    Matrix3r d_axis_normal_d_n1 = Matrix3r::Zero();
    Matrix3r d_axis_normal_d_n2 = Matrix3r::Zero();
    if (sin_angle > 1e-8 * normal.norm() * other_normal.norm()) {
        axis_normal = normal.cross(other_normal) / sin_angle;
        d_axis_normal_d_n1 = -CrossProductMatrix(other_normal) / sin_angle + (axis_normal * axis_normal.transpose()) * CrossProductMatrix(other_normal) / sin_angle;
        d_axis_normal_d_n2 = CrossProductMatrix(normal) / sin_angle - (axis_normal * axis_normal.transpose()) * CrossProductMatrix(normal) / sin_angle;
    }
    const Vector3r d_sin_d_n1 = other_normal.cross(axis_normal);
    const Vector3r d_sin_d_n2 = -normal.cross(axis_normal);
    
    const Vector3r d_cos_d_n1 = other_normal;
    const Vector3r d_cos_d_n2 = normal;

    const Vector3r d_sclae_d_n1 = 2 * cos_angle * d_cos_d_n1 + 2 * sin_angle * d_sin_d_n1;
    const Vector3r d_sclae_d_n2 = 2 * cos_angle * d_cos_d_n2 + 2 * sin_angle * d_sin_d_n2;
    const Vector3r d_d_angle_d_sin_d_n1 = (d_cos_d_n1 * scale - cos_angle * d_sclae_d_n1) / (scale * scale);
    const Vector3r d_d_angle_d_sin_d_n2 = (d_cos_d_n2 * scale - cos_angle * d_sclae_d_n2) / (scale * scale);
    const Vector3r d_d_angle_d_cos_d_n1 = -(d_sin_d_n1 * scale - sin_angle * d_sclae_d_n1) / (scale * scale);
    const Vector3r d_d_angle_d_cos_d_n2 = -(d_sin_d_n2 * scale - sin_angle * d_sclae_d_n2) / (scale * scale);

    const Matrix3r d_d_sin_d_n1_d_n1 = CrossProductMatrix(other_normal) * d_axis_normal_d_n1;
    const Matrix3r d_d_sin_d_n1_d_n2 = CrossProductMatrix(other_normal) * d_axis_normal_d_n2 - CrossProductMatrix(axis_normal);
    const Matrix3r d_d_sin_d_n2_d_n1 = -CrossProductMatrix(normal) * d_axis_normal_d_n1 + CrossProductMatrix(axis_normal);
    const Matrix3r d_d_sin_d_n2_d_n2 = -CrossProductMatrix(normal) * d_axis_normal_d_n2;

    // const Vector3r d_angle_d_n1 = d_angle_d_sin * d_sin_d_n1 + d_angle_d_cos * d_cos_d_n1;
    // const Vector3r d_angle_d_n2 = d_angle_d_sin * d_sin_d_n2 + d_angle_d_cos * d_cos_d_n2;
    const Matrix3r d_d_angle_d_n1_d_n1 = d_angle_d_sin * d_d_sin_d_n1_d_n1 + d_sin_d_n1 * d_d_angle_d_sin_d_n1.transpose() + d_cos_d_n1 * d_d_angle_d_cos_d_n1.transpose();
    const Matrix3r d_d_angle_d_n1_d_n2 = d_angle_d_sin * d_d_sin_d_n1_d_n2 + d_sin_d_n1 * d_d_angle_d_sin_d_n2.transpose() + d_cos_d_n1 * d_d_angle_d_cos_d_n2.transpose() + d_angle_d_cos * Matrix3r::Identity();
    const Matrix3r d_d_angle_d_n2_d_n1 = d_angle_d_sin * d_d_sin_d_n2_d_n1 + d_sin_d_n2 * d_d_angle_d_sin_d_n1.transpose() + d_cos_d_n2 * d_d_angle_d_cos_d_n1.transpose() + d_angle_d_cos * Matrix3r::Identity();
    const Matrix3r d_d_angle_d_n2_d_n2 = d_angle_d_sin * d_d_sin_d_n2_d_n2 + d_sin_d_n2 * d_d_angle_d_sin_d_n2.transpose() + d_cos_d_n2 * d_d_angle_d_cos_d_n2.transpose();

    hess[0] = d_d_angle_d_n1_d_n1;
    hess[1] = d_d_angle_d_n1_d_n2;
    hess[2] = d_d_angle_d_n2_d_n1;
    hess[3] = d_d_angle_d_n2_d_n2;

    return hess;
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
            if (info.other_triangle == -1) continue;
            const Vector3r other_normal = ComputeNormal(position(Eigen::all, elements_.col(info.other_triangle)));
            const real angle = ComputeDihedralAngleFromNonUnitNormal(normal, other_normal);
            const real rest_shape_edge_length = info.edge_length;
            const real diamond_area = 0.01 / 3;
            // TODO.
            energy += (angle * rest_shape_edge_length) * (angle * rest_shape_edge_length) / diamond_area;
            /////////////////////////////////////////////
        }
    }
    return bending_stiffness_ * energy;
}

const Matrix3Xr Simulator::ComputeBendingForce(const Matrix3Xr& position) const {
    // TODO.
    Matrix3Xr grad = Matrix3Xr::Zero(3, position.cols());
    grad.setZero();
    const integer element_num = static_cast<integer>(elements_.cols());
    for (integer e = 0; e < element_num; e++) {
        const Matrix3r vertices = position(Eigen::all, elements_.col(e));
        const Vector3r normal = ComputeNormal(vertices);
        const Matrix3Xr normal_gradient = ComputeNormalGradient(vertices);
        for (integer i = 0; i < 3; i++) {
            const TriangleEdgeInfo& info = triangle_edge_info_[e][i];
            if (info.other_triangle == -1) continue;
            const Matrix3r other_vertices = position(Eigen::all, elements_.col(info.other_triangle));
            const Vector3r other_normal = ComputeNormal(other_vertices);
            const real angle = ComputeDihedralAngleFromNonUnitNormal(normal, other_normal);
            const real rest_shape_edge_length = info.edge_length;
            const real diamond_area = 0.01 / 3;

            const Matrix3Xr other_normal_gradient = ComputeNormalGradient(other_vertices);
            const auto [d_angle_d_normal, d_angle_d_other_normal] = ComputeDihedralAngleGradient(normal, other_normal);
            const Matrix3r d_angle_d_v = (d_angle_d_normal.transpose() * normal_gradient).reshaped(3, 3);
            const Matrix3r d_angle_d_other_v = (d_angle_d_other_normal.transpose() * other_normal_gradient).reshaped(3, 3);

            const real coefficient = 2 * bending_stiffness_ * (angle * rest_shape_edge_length) * rest_shape_edge_length / diamond_area;

            for (integer j = 0; j < 3; j++) {
                grad.col(elements_(j, e)) += coefficient * d_angle_d_v.col(j);
                grad.col(elements_(j, info.other_triangle)) += coefficient * d_angle_d_other_v.col(j);
            }
        }
    }
    return -grad;
    /////////////////////////////////////
}

const SparseMatrixXr Simulator::ComputeBendingHessian(const Matrix3Xr& position) const {
    // TODO.
    const integer hess_size = static_cast<integer>(position.cols()) * 3;
    const integer element_num = static_cast<integer>(elements_.cols());
    std::vector<Matrix9r> hess_non_zeros;
    std::vector<std::pair<integer, integer>> hess_indices;
    for (integer e = 0; e < element_num; ++e) {
        const Matrix3r vertices = position(Eigen::all, elements_.col(e));
        const Vector3r normal = ComputeNormal(vertices);
        const Matrix3Xr normal_gradient = ComputeNormalGradient(vertices);
        const auto normal_hessian = ComputeNormalHessian(vertices);
        for (integer i = 0; i < 3; ++i) {
            const TriangleEdgeInfo& info = triangle_edge_info_[e][i];
            if (info.other_triangle == -1) continue;
            const Matrix3r other_vertices = position(Eigen::all, elements_.col(info.other_triangle));
            const Vector3r other_normal = ComputeNormal(other_vertices);
            const Matrix3Xr other_normal_gradient = ComputeNormalGradient(other_vertices);

            const real angle = ComputeDihedralAngleFromNonUnitNormal(normal, other_normal);
            const auto [d_angle_d_n1, d_angle_d_n2] = ComputeDihedralAngleGradient(normal, other_normal);
            const auto d_angle_hessian = ComputeDihedralAngleHessian(normal, other_normal);
            const Matrix3r d_angle_d_v1 = (d_angle_d_n1.transpose() * normal_gradient).reshaped(3, 3);
            const Matrix3r d_angle_d_v2 = (d_angle_d_n2.transpose() * other_normal_gradient).reshaped(3, 3);

            Matrix9r d_d_angle_d_v1_d_v1 = normal_gradient.transpose() * d_angle_hessian[0] * normal_gradient;
            Matrix9r d_d_angle_d_v1_d_v2 = normal_gradient.transpose() * d_angle_hessian[1] * other_normal_gradient;
            Matrix9r d_d_angle_d_v2_d_v1 = other_normal_gradient.transpose() * d_angle_hessian[2] * normal_gradient;
            Matrix9r d_d_angle_d_v2_d_v2 = other_normal_gradient.transpose() * d_angle_hessian[3] * other_normal_gradient;
            for (integer j = 0; j < 9; j++) {
                d_d_angle_d_v1_d_v1.col(j) += Vector9r(d_angle_d_n1.transpose() * normal_hessian[j]);
                d_d_angle_d_v2_d_v2.col(j) += Vector9r(d_angle_d_n2.transpose() * normal_hessian[j]);
            }

            const real rest_shape_edge_length = info.edge_length;
            const real diamond_area = 0.01 / 3;
            const real coefficient = 2 * bending_stiffness_ * rest_shape_edge_length * rest_shape_edge_length / diamond_area;

            const Vector9r d_angle_d_v1_flatten = d_angle_d_v1.reshaped();
            const Vector9r d_angle_d_v2_flatten = d_angle_d_v2.reshaped();
            const Matrix9r hess_v1_v1 = coefficient * (d_angle_d_v1_flatten * d_angle_d_v1_flatten.transpose() + d_d_angle_d_v1_d_v1 * angle);
            const Matrix9r hess_v2_v2 = coefficient * (d_angle_d_v2_flatten * d_angle_d_v2_flatten.transpose() + d_d_angle_d_v2_d_v2 * angle);
            const Matrix9r hess_v1_v2 = coefficient * (d_angle_d_v1_flatten * d_angle_d_v2_flatten.transpose() + d_d_angle_d_v1_d_v2 * angle);
            const Matrix9r hess_v2_v1 = coefficient * (d_angle_d_v2_flatten * d_angle_d_v1_flatten.transpose() + d_d_angle_d_v2_d_v1 * angle);

            hess_non_zeros.push_back(hess_v1_v1);
            hess_indices.push_back({ e, e });
            hess_non_zeros.push_back(hess_v2_v2);
            hess_indices.push_back({ info.other_triangle, info.other_triangle });
            hess_non_zeros.push_back(hess_v1_v2);
            hess_indices.push_back({ e, info.other_triangle });
            hess_non_zeros.push_back(hess_v2_v1);
            hess_indices.push_back({ info.other_triangle, e });

        }
    }
    SparseMatrixXr hess(hess_size, hess_size);
    hess.setZero();
    const integer hess_non_zeros_num = static_cast<integer>(hess_non_zeros.size());
    for (integer i = 0; i < hess_non_zeros_num; ++i) {
        const auto& hess_non_zero = hess_non_zeros[i];
        const auto& hess_index = hess_indices[i];
        const Vector3i row_indices = elements_.col(hess_index.first);
        const Vector3i col_indices = elements_.col(hess_index.second);
        for (integer j = 0; j < 3; ++j) {
            for (integer k = 0; k < 3; ++k) {
                for (integer dj = 0; dj < 3; dj++) {
                    for (integer dk = 0; dk < 3; dk++) {
                        hess.coeffRef(row_indices(j) * 3 + dj, col_indices(k) * 3 + dk) += hess_non_zero(j * 3 + dj, k * 3 + dk);
                    }
                }
            }
        }
    }

    return hess;
    /////////////////////////////////////
}

}
}