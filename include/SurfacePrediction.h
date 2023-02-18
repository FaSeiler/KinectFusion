#pragma once
#include <memory>
#include "Ray.h"
#include "Eigen/Dense"
#include "Volume.h"
#include "SurfaceMeasurement.h"
#include "VirtualSensor.h"


namespace SurfacePrediction
{
    void  FillRay(Ray* ray, const Eigen::Vector3d& origin, const Eigen::Vector3d& direction);
    // Predicts surface, modifies the current frame with the updated info
    void surfacePrediction(CameraParametersPyramid cameraParams, std::shared_ptr<SurfaceMeasurement>& currentFrame, std::shared_ptr<Volume>& volume,
        float truncationDistance, int numLevels);

    // Returns the normalized direction of ray
    Eigen::Vector3d calculateRayDirection(size_t x, size_t y, const Eigen::Matrix3d& rotation,
        CameraParametersPyramid cameraParams);

    // useful for calculating normal
    double TSDFInterpolation(const Eigen::Vector3d& gridVertex, const std::shared_ptr<Volume>& volume);

    Eigen::Vector3d getVertexAtZeroCrossing(
        const Eigen::Vector3d& prevPoint, const Eigen::Vector3d& currPoint,
        double prevTSDF, double currTSDF);

    // calculate normal at vertex
    Eigen::Vector3d calculateNormal(const Eigen::Vector3d& gridVertex, const std::shared_ptr<Volume>& volume, float truncationDistance);
    // is within bounds
    bool isInBounds(const Eigen::Vector3d& gridVertex, Eigen::Vector3i volumeSize);

};