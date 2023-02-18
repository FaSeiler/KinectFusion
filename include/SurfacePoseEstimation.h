#pragma once

#include <memory>

#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/StdVector>

#include <vector>
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include "NearestNeighbor.h"
#include "Mesh.h"
#include "SurfaceMeasurement.h"
#include "SurfacePoseEstimationCuda.h"
#include "iterator"

#define USE_CUDA_SURFACE_POSE_ESTIMATION 0

using namespace std;
using namespace Eigen;
using namespace ceres;

class SurfacePoseEstimation {
public:
    SurfacePoseEstimation();

    bool EstimateTransform(std::shared_ptr<SurfaceMeasurement> curr_frame, std::shared_ptr<SurfaceMeasurement> prev_frame, Matrix4d& startTransform);
    void SetKNNDistance(float maxDistance);
    void SetNrIterations(unsigned nrIterations);
    void LogCeresUpdates(bool logCeresOptimizerStep, bool logCeresFullReport);

private:
    Eigen::Matrix<double, 6, 1> solution;

    Matrix3d GetRotationFromTransform(const Matrix4d& transform);
    Vector3d GetTranslationFromTransform(const Matrix4d& transform);
    vector<Vector3d> ApplyTransformToPoints(const vector<Vector3d>& sourcePoints, const Matrix4d& transform);
    vector<Vector3d> ApplyTransformToNormals(const vector<Vector3d>& sourceNormals, const Matrix4d& transform);
    static Matrix4d TransformArrayToMatrix(double& transformArray);

    void Pruning(const vector<Vector3d>& sourceNormals, const vector<Vector3d>& targetNormals, vector<Match>& matches);
    void SetConstraints(const vector<Vector3d>& sourcePoints, const vector<Vector3d>& targetPoints, const vector<Vector3d>& targetNormals,
        const vector<Match> matches, const double& transformArray, Problem& problem);

    bool m_logCeresOptimizerStep;
    bool m_logFullReport;
    unsigned m_nrIterations;
    unique_ptr<NearestNeighborSearch> m_nearestNeighborSearch;

    class PointToPointCostFunctor {
    public:
        PointToPointCostFunctor(const Vector3d& sourcePoint, const Vector3d& targetPoint, const float weight) :
            m_pSource{ sourcePoint },
            m_pTarget{ targetPoint },
            m_weight{ weight }
        { }

        template <typename T>
        bool operator()(const T* const transform, T* residuals) const {
            T pSourceInp[3];
            pSourceInp[0] = T(m_pSource[0]);
            pSourceInp[1] = T(m_pSource[1]);
            pSourceInp[2] = T(m_pSource[2]);

            T pSourceOut[3];
            const T* rotation = transform;
            const T* translation = transform + 3;
            // Apply rotation
            T pSourceInpRotated[3];
            ceres::AngleAxisRotatePoint(rotation, pSourceInp, pSourceInpRotated);
            // Apply translation
            pSourceOut[0] = pSourceInpRotated[0] + translation[0];
            pSourceOut[1] = pSourceInpRotated[1] + translation[1];
            pSourceOut[2] = pSourceInpRotated[2] + translation[2];

            T pTarget[3];
            pTarget[0] = T(m_pTarget[0]);
            pTarget[1] = T(m_pTarget[1]);
            pTarget[2] = T(m_pTarget[2]);

            // M * ps - pt 
            residuals[0] = (pSourceOut[0] - pTarget[0]) * T(m_weight) * T(m_lambda);
            residuals[1] = (pSourceOut[1] - pTarget[1]) * T(m_weight) * T(m_lambda);
            residuals[2] = (pSourceOut[2] - pTarget[2]) * T(m_weight) * T(m_lambda);

            return true;
        }

        static CostFunction* PointToPointCostFunction(const Vector3d& sourcePoint, const Vector3d& targetPoint, const float weight) {
            return new AutoDiffCostFunction<PointToPointCostFunctor, 3, 6>(
                new PointToPointCostFunctor(sourcePoint, targetPoint, weight)
                );
        }

    protected:
        const Vector3d m_pSource;
        const Vector3d m_pTarget;
        const float m_weight;
        const float m_lambda = 0.1f;
    };

    class PointToPlaneCostFunctor {
    public:
        PointToPlaneCostFunctor(const Vector3d& sourcePoint, const Vector3d& targetPoint, const Vector3d& targetNormal) :
            m_pSource{ sourcePoint },
            m_pTarget{ targetPoint },
            m_nTarget{ targetNormal }
        { }

        template <typename T>
        bool operator()(const T* const transform, T* residuals) const {

            // Init the source point
            T pSourceInp[3];
            pSourceInp[0] = T(m_pSource[0]);
            pSourceInp[1] = T(m_pSource[1]);
            pSourceInp[2] = T(m_pSource[2]);

            // Update the point using the current transform
            T pSourceOut[3];
            const T* rotation = transform;
            const T* translation = transform + 3;
            // Apply rotation
            T pSourceInpRotated[3];
            ceres::AngleAxisRotatePoint(rotation, pSourceInp, pSourceInpRotated);
            // Apply translation
            pSourceOut[0] = pSourceInpRotated[0] + translation[0];
            pSourceOut[1] = pSourceInpRotated[1] + translation[1];
            pSourceOut[2] = pSourceInpRotated[2] + translation[2];

            // Init the target point
            T pTarget[3];
            pTarget[0] = T(m_pTarget[0]);
            pTarget[1] = T(m_pTarget[1]);
            pTarget[2] = T(m_pTarget[2]);

            // Init the target normal
            T nTarget[3];
            nTarget[0] = T(m_nTarget[0]);
            nTarget[1] = T(m_nTarget[1]);
            nTarget[2] = T(m_nTarget[2]);

            // Define residual function
            // (M * ps - pt) * n
            residuals[0] =
                (nTarget[0] * (pSourceOut[0] - pTarget[0])
                    + nTarget[1] * (pSourceOut[1] - pTarget[1])
                    + nTarget[2] * (pSourceOut[2] - pTarget[2]));

            return true;
        }

        static CostFunction* PointToPlaneCostFunction(const Vector3d& sourcePoint, const Vector3d& targetPoint, const Vector3d& targetNormal) {
            return new AutoDiffCostFunction<PointToPlaneCostFunctor, 1, 6>( // 1 = dimension of residual, 6 = dimension of transform (3 transl, 3 rot)
                new PointToPlaneCostFunctor(sourcePoint, targetPoint, targetNormal));
        }

    private:
        const Vector3d m_pSource;
        const Vector3d m_pTarget;
        const Vector3d m_nTarget;
    };
};