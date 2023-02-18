#pragma once


#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "Volume.h"
#include <SurfaceMeasurement.h>
#include <memory>
#include "VirtualSensor.h"

namespace SurfaceReconstructionCuda {
    namespace Volume {
        __device__ Eigen::Vector3d getGlobalCoordinate(::Volume* volume, int voxelIdx_x, int voxelIdx_y, int voxelIdx_z);
     }

    namespace SurfaceMeasurement {
        __device__ Eigen::Vector2i projectOntoPlane(const Eigen::Vector3d& cameraCoord, Eigen::Matrix3d& intrinsics, float mInf);
        __device__ Eigen::Vector2i projectOntoDepthPlane(const Eigen::Vector3d& cameraCoord, CameraParametersPyramid cameraParams, float mInf);
        __device__ Eigen::Matrix3d intrinsicPyramid(CameraParametersPyramid cameraParams);
        __device__ bool containsCuda(const Eigen::Vector2i& img_coord, CameraParametersPyramid cameraParams);


    }
    //THIS method expects frame to hold all camera paramerters as well as the estimated pose --> TODO: check if those values are set or redefine method parameters

    void reconstructSurface(CameraParametersPyramid cameraParams, std::shared_ptr<::SurfaceMeasurement>& currentFrame, const std::shared_ptr<::Volume>& volume, double truncationDistance, int numLevels);
    __global__ void reconstructSurfaceKernel(size_t x, size_t y, size_t z, size_t depthColumns,
        Eigen::Matrix3d rotation, Eigen::Vector3d translation, double truncationDistance, float* depthMap,
        ::Volume* volume, Voxel* voxels, CameraParametersPyramid& cameraParams, ::SurfaceMeasurement* currentFrame,
        Eigen::Matrix<unsigned char, 4, 1>* colors);


    /*!
         * The original implementation actually takes a raw depth Value, as we already calculated the camereSpacePoints
         * only the normalization has to be done.
         * TODO: move normalization to frame.cpp ; check if the cameraSpaceTransformation in frame.cpp equals the one used in Paper
         * @param cameraSpacePoint
         * @return the normalized cameraSpaceCoordinates
         */
    __device__ double calculateLamdas(Eigen::Vector2i& cameraSpacePoint, CameraParametersPyramid cameraParams);
    /*!
     *
     * @param lambda
     * @param cameraPosition
     * @param rawDepthValue
     * @return the signed-distance-function for the specific depth value lambda is based on
     */

    __device__ double calculateSDF(double& lambda, Eigen::Vector3d& cameraPosition, double rawDepthValue);

};
