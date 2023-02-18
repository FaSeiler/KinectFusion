#include "SurfacePoseEstimation.h"

SurfacePoseEstimation::SurfacePoseEstimation() {
    m_logCeresOptimizerStep = true;
    m_logFullReport = false;
    m_nrIterations = 15;
    m_nearestNeighborSearch = make_unique<NearestNeighborSearch>();
}

vector<Match> CUDA_NN_Interface(vector<Vector3d> source, vector<Vector3d> target) {
    int data_pts = target.size();
    int query_pts = source.size();

    vector <float> data(data_pts * CUDA_NN_DIM);
    vector <float> query(query_pts * CUDA_NN_DIM);

    for (size_t pointIndex = 0; pointIndex < query_pts; pointIndex++) {
        for (size_t dim = 0; dim < 3; dim++) {
            query[pointIndex * 3 + dim] = source[pointIndex][dim];
        }
    }

    for (size_t pointIndex = 0; pointIndex < data_pts; pointIndex++) {
        for (size_t dim = 0; dim < 3; dim++) {
            data[pointIndex * 3 + dim] = target[pointIndex][dim];
        }
    }

    vector <int> idxs(query_pts);
    vector <float> dist_sq(query_pts);
    vector <int> cpu_idxs(query_pts);
    vector <float> cpu_dist_sq(query_pts);

    Cuda::CUDA_NN_Search(&query[0], query_pts, &data[0], data_pts, &idxs[0], &dist_sq[0]);

    const unsigned nMatches = query_pts;
    vector<Match> matches;
    matches.reserve(nMatches);
    float m_maxDistance = 0.0001f;
    for (int i = 0; i < nMatches; ++i) {
        float tmp = dist_sq[i];
        if (dist_sq[i] <= m_maxDistance)
            matches.push_back(Match{ idxs[i], 1.f });
        else
            matches.push_back(Match{ -1, 0.f });
    }
    return matches;
}

bool SurfacePoseEstimation::EstimateTransform(std::shared_ptr<SurfaceMeasurement> curr_frame, std::shared_ptr<SurfaceMeasurement> prev_frame, Matrix4d& startTransform) {
    Matrix4d estimatedTransform = startTransform;
    double transformArray[6] = { 0., 0., 0., 0., 0., 0. }; // [0...2] rotation, [3...5] translation
    m_nearestNeighborSearch->buildIndex(prev_frame->getVertexMap());

    for (int iteration = 0; iteration < m_nrIterations; ++iteration) {
        clock_t begin = clock();

        auto transformedPoints = ApplyTransformToPoints(curr_frame->getVertexMap(), estimatedTransform);
        auto transformedNormals = ApplyTransformToNormals(curr_frame->getNormalMap(), estimatedTransform);

        auto matches = m_nearestNeighborSearch->queryMatches(transformedPoints);
        //auto matches = CUDA_NN_Interface(curr_frame->getVertexMap(), transformedPoints); // IF USE_CUDA
        Pruning(transformedNormals, prev_frame->getNormalMap(), matches);

        Problem problem;
        SetConstraints(transformedPoints, prev_frame->getVertexMap(), prev_frame->getNormalMap(), matches, *transformArray, problem);

        
        Solver::Options options;
#if USE_CUDA_SURFACE_POSE_ESTIMATION
        options.dense_linear_algebra_library_type = CUDA;
#else
        options.linear_solver_type = ceres::DENSE_QR;
#endif
        options.num_threads = 8; // 8 threads is max
        options.max_num_iterations = 1;
        options.minimizer_progress_to_stdout = m_logCeresOptimizerStep;

        Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);

        if (m_logFullReport)
        {
            cout << summary.FullReport() << endl;
        }

        Matrix4d matrix = TransformArrayToMatrix(*transformArray);
        estimatedTransform = matrix * estimatedTransform;

        for (size_t i = 0; i < 6; i++)
        {
            transformArray[i] = 0;
        }

        clock_t end = clock();
        double deltaTime = double(end - begin) / CLOCKS_PER_SEC;
        cout << "[" << iteration << " | " << m_nrIterations - 1 << "] " << "Optimization step: " << deltaTime << " seconds." << endl;
    }

    cout << "=============================================================" << endl;
    startTransform = estimatedTransform;
    curr_frame->setGlobalPose(estimatedTransform);

    return true;
}

void SurfacePoseEstimation::LogCeresUpdates(bool logCeresOptimizerStep, bool logCeresFullReport) {
    m_logCeresOptimizerStep = logCeresOptimizerStep;
    m_logFullReport = logCeresFullReport;
}

void SurfacePoseEstimation::SetKNNDistance(float maxDistance) {
    m_nearestNeighborSearch->SetMatchingDistance(maxDistance);
}

void SurfacePoseEstimation::SetNrIterations(unsigned nrIterations) {
    m_nrIterations = nrIterations;
}

Matrix3d SurfacePoseEstimation::GetRotationFromTransform(const Matrix4d& transform) {
    return transform.block(0, 0, 3, 3);
}

Vector3d SurfacePoseEstimation::GetTranslationFromTransform(const Matrix4d& transform) {
    return transform.block(0, 3, 3, 1);
}

vector<Vector3d> SurfacePoseEstimation::ApplyTransformToPoints(const vector<Vector3d>& sourcePoints, const Matrix4d& transform) {
    vector<Vector3d> transformedPoints;
    const auto rotation = GetRotationFromTransform(transform);
    const auto translation = GetTranslationFromTransform(transform);

    for (const Vector3d& point : sourcePoints) {
        transformedPoints.push_back(rotation * point + translation);
    }

    return transformedPoints;
}

vector<Vector3d> SurfacePoseEstimation::ApplyTransformToNormals(const vector<Vector3d>& sourceNormals, const Matrix4d& transform) {
    vector<Vector3d> transformedNormals;
    const auto rotation = GetRotationFromTransform(transform);

    for (const Vector3d& normal : sourceNormals) {
        transformedNormals.push_back(rotation.inverse().transpose() * normal);
    }

    return transformedNormals;
}

void SurfacePoseEstimation::Pruning(const vector<Vector3d>& sourceNormals, const vector<Vector3d>& targetNormals, vector<Match>& matches) {

    for (size_t i = 0; i < sourceNormals.size(); i++) {
        Match& match = matches[i];
        if (match.idx >= 0) {
            const auto& sourceNormal = sourceNormals[i];
            const auto& targetNormal = targetNormals[match.idx];

            double angleRad = acos(sourceNormal.dot(targetNormal) / (sourceNormal.norm() * targetNormal.norm()));
            double angleGrad = angleRad * 180 / M_PI;

            if (angleGrad > 60) {
                match.idx = -1;
            }
        }
    }
}

void SurfacePoseEstimation::SetConstraints(const vector<Vector3d>& sourcePoints, const vector<Vector3d>& targetPoints, const vector<Vector3d>& targetNormals,
    const vector<Match> matches, const double& transformArray, Problem& problem) {

    for (unsigned i = 0; i < sourcePoints.size(); ++i) {
        const Match match = matches[i];
        if (match.idx >= 0) {
            const auto& sourcePoint = sourcePoints[i];
            const auto& targetPoint = targetPoints[match.idx];
            const auto& targetNormal = targetNormals[match.idx];

            // Check if all values in sourcePoint are not NaN
            if (!sourcePoint.allFinite() || !targetPoint.allFinite() || !targetNormal.allFinite())
                continue;

            problem.AddResidualBlock(
                PointToPointCostFunctor::PointToPointCostFunction(sourcePoint, targetPoint, 1.0),
                nullptr, const_cast<double*>(&transformArray)
            );

            problem.AddResidualBlock(
                PointToPlaneCostFunctor::PointToPlaneCostFunction(sourcePoint, targetPoint, targetNormal),
                nullptr, const_cast<double*>(&transformArray)
            );

        }
    }
}

Matrix4d SurfacePoseEstimation::TransformArrayToMatrix(double& transformArray) {
    // transformArray = { r1, r2, r3, t1, t2, t3 }
    double* rotation = &transformArray;
    double* translation = &transformArray + 3;

    double rotationMatrix[9];
    ceres::AngleAxisToRotationMatrix(rotation, rotationMatrix);

    Matrix4d transformMatrix;
    transformMatrix.setIdentity();
    transformMatrix(0, 0) = float(rotationMatrix[0]);
    transformMatrix(1, 0) = float(rotationMatrix[1]);
    transformMatrix(2, 0) = float(rotationMatrix[2]);

    transformMatrix(0, 1) = float(rotationMatrix[3]);
    transformMatrix(1, 1) = float(rotationMatrix[4]);
    transformMatrix(2, 1) = float(rotationMatrix[5]);

    transformMatrix(0, 2) = float(rotationMatrix[6]);
    transformMatrix(1, 2) = float(rotationMatrix[7]);
    transformMatrix(2, 2) = float(rotationMatrix[8]);

    transformMatrix(0, 3) = float(translation[0]);
    transformMatrix(1, 3) = float(translation[1]);
    transformMatrix(2, 3) = float(translation[2]);

    return transformMatrix;
}