#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <memory>
#include "Ray.h"
#include "Eigen/Dense"
#include "Volume.h"
#include "SurfaceMeasurement.h"
#include "VirtualSensor.h"

#include <stdio.h>

namespace Cuda {
    namespace Volume {
        __device__ bool intersects(Vector3d* boundsCuda, Ray* r, float entry_distance);
        __device__ bool contains(Vector3d* _originCuda, Vector3d* _volumeRangeCuda, const Eigen::Vector3d global_point);
        __device__ Eigen::Vector3d  getGlobalCoordinate(::Volume* volume, int voxelIdx_x, int voxelIdx_y, int voxelIdx_z);
        __device__ Vector4uc getColor(::Volume* volume, Eigen::Vector3d global);
        __device__ double getTSDF(::Volume* volume, Eigen::Vector3d global);
        __device__ double getTSDF(::Volume* volume, int x, int y, int z);

    };
    namespace SurfacePrediction
    {
        __device__ static float mInf = -std::numeric_limits<float>::infinity();
        // Predicts surface, modifies the current frame with the updated info
        void surfacePrediction(CameraParametersPyramid cameraParams, std::shared_ptr<SurfaceMeasurement> currentFrame, std::shared_ptr<::Volume> volume, float truncationDistance, const int numLevels);
        __global__ void surfacePrediction_kernel(size_t width, size_t height,
            Eigen::Vector3i* volumeSize,  Eigen::Vector3d* volumeRange, Eigen::Vector3d* volumeOrigin, double voxelScale,
            Eigen::Matrix4d* pose, Eigen::Block<Eigen::Matrix4d>* rotationMatrix, Eigen::Block<Eigen::Matrix4d>* translation, CameraParametersPyramid* cameraParams,
            Eigen::Vector3d* vertices, Eigen::Matrix<unsigned char, 4, 1>* colors, float truncationDistance, int numLevels, Vector3d* boundsCuda);
        // Returns the normalized direction of ray
        __device__ Eigen::Vector3d  calculateRayDirection(size_t x, size_t y, const Eigen::Matrix3d& rotation,
            CameraParametersPyramid cameraParams);

        // useful for calculating normal
        __device__ double TSDFInterpolation(const Eigen::Vector3d& gridVertex, ::Volume& volume);

        __device__ Eigen::Vector3d getVertexAtZeroCrossing(
            const Eigen::Vector3d& prevPoint, const Eigen::Vector3d& currPoint,
            const double prevTSDF, const double currTSDF);

        // calculate normal at vertex
        __device__ Eigen::Vector3d calculateNormal(Eigen::Vector3d& gridVertex, ::Volume& volume);
        // is within bounds
        __device__ bool isInBounds(const Eigen::Vector3d& gridVertex, Eigen::Vector3i volumeSize);
        __device__  void FillRay(Ray* ray, const Eigen::Vector3d& origin, const Eigen::Vector3d& direction);


    };

}