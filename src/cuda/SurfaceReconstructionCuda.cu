#include <iostream>
#include "SurfaceReconstructionCuda.h"
#include "SurfacePredictionCuda.h"
namespace SurfaceReconstructionCuda {
    namespace Volume{
        __device__ Eigen::Vector3d Volume::getGlobalCoordinate(::Volume* volume, int voxelIdx_x, int voxelIdx_y, int voxelIdx_z) {
            const Eigen::Vector3d position((static_cast<double>(voxelIdx_x) + 0.5) * *volume->_voxelScaleCuda,
                (static_cast<double>(voxelIdx_y) + 0.5) * *volume->_voxelScaleCuda,
                (static_cast<double>(voxelIdx_z) + 0.5) * *volume->_voxelScaleCuda);
            return position + *volume->_originCuda;
        }
    }
    namespace SurfaceMeasurement {
        __device__ Eigen::Vector2i SurfaceMeasurement::projectOntoPlane(const Eigen::Vector3d& cameraCoord, Eigen::Matrix3d& intrinsics, float mInf) {
            Eigen::Vector3d projected = (intrinsics * cameraCoord);
            if (projected[2] == 0) {
                return Eigen::Vector2i(mInf, mInf);
            }
            projected /= projected[2];
            return (Eigen::Vector2i((int)round(projected.x()), (int)round(projected.y())));
        }
        __device__ Eigen::Vector2i SurfaceMeasurement::projectOntoDepthPlane( const Eigen::Vector3d& cameraCoord, CameraParametersPyramid cameraParams, float mInf) {
            auto intrinsics = intrinsicPyramid(cameraParams);

            return projectOntoPlane(cameraCoord, intrinsics, mInf);
        }

        __device__ Eigen::Matrix3d SurfaceMeasurement::intrinsicPyramid(CameraParametersPyramid cameraParams) {

            Eigen::Matrix3d intrinsics;
            intrinsics << cameraParams.fovX, 0.0f, cameraParams.cX, 0.0f, cameraParams.fovY, cameraParams.cY, 0.0f, 0.0f, 1.0f;

            return intrinsics;
        }

        __device__ bool SurfaceMeasurement::containsCuda(const Eigen::Vector2i& img_coord, CameraParametersPyramid cameraParams) {
            return img_coord[0] < cameraParams.imageWidth && img_coord[1] < cameraParams.imageHeight && img_coord[0] >= 0 && img_coord[1] >= 0;
        }

    }
    void reconstructSurface(CameraParametersPyramid cameraParams, std::shared_ptr<::SurfaceMeasurement>& currentFrame,
        const std::shared_ptr<::Volume>& volume, double truncationDistance, int numLevels) {

        auto volumeSize = volume->getVolumeSize();
        auto pose = currentFrame->getGlobalPose().inverse();
        auto voxelData = volume->getVoxelData();

        cv::Mat depthMap = currentFrame->getDepthMapPyramid();

        Eigen::Matrix3d rotation = pose.block(0, 0, 3, 3);
        Eigen::Vector3d translation = pose.block(0, 3, 3, 1);

        size_t x = volumeSize.x();
        size_t y = volumeSize.y();
        size_t z = volumeSize.z();
        dim3 numBlocks((x + 7)/ 8, (y + 7 )/ 8, (z + 7 ) / 8);
        dim3 numThreads(8, 8, 8);

        Eigen::Matrix<unsigned char, 4, 1>* colors;

        size_t colorSize = currentFrame->getColorMap().size();
        std::vector<float> depthArray;
        if (depthMap.isContinuous()) {
            depthArray.assign((float*)depthMap.data, (float*)depthMap.data + depthMap.total() * depthMap.channels());
        }
        for (int i = 0; i < depthMap.rows; ++i) {
            depthArray.insert(depthArray.end(), depthMap.ptr<float>(i), depthMap.ptr<float>(i) + depthMap.cols * depthMap.channels());
        }
        float* depthArrayCuda;

        dim3 size = dim3(volumeSize.x(), volumeSize.y(), volumeSize.z());
        cudaMalloc(&colors, colorSize * sizeof(Eigen::Matrix<unsigned char, 4, 1>));
        cudaMalloc(&depthArrayCuda, depthArray.size() * sizeof(float));
        
        cudaMemcpy(colors, currentFrame->getColorMap().data(), colorSize * sizeof(Eigen::Matrix<unsigned char, 4, 1>), cudaMemcpyHostToDevice);
        cudaMemcpy(depthArrayCuda, depthArray.data(), depthArray.size() * sizeof(float), cudaMemcpyHostToDevice);

        reconstructSurfaceKernel<< <numBlocks, numThreads >> > (x, y, z, depthMap.cols,
            rotation, translation, truncationDistance, depthArrayCuda,
            volume.get(), volume->_voxelDataCuda, cameraParams.level(numLevels), currentFrame.get(),
            colors);

        //cudaDeviceSynchronize();


        cudaFree(colors);
        cudaFree(depthArrayCuda);
    }

    __global__ void reconstructSurfaceKernel(size_t x, size_t y, size_t z, size_t depthColumns,
        Eigen::Matrix3d rotation, Eigen::Vector3d translation, double truncationDistance, float* depthMap,
        ::Volume* volume, Voxel* voxels, CameraParametersPyramid& cameraParams, ::SurfaceMeasurement* currentFrame, 
        Eigen::Matrix<unsigned char, 4, 1>* colors) {


        int indexX = blockIdx.x * blockDim.x + threadIdx.x;
        int indexY = blockIdx.y * blockDim.y + threadIdx.y;
        int indexZ = blockIdx.z * blockDim.z + threadIdx.z;

        if (indexX < x && indexY < y && indexZ < z) {
          
            /*
            * Volumetric Reconstruction
            */
            //calculate Camera Position
            Eigen::Vector3d globalCoord_voxel = Volume::getGlobalCoordinate(volume, x, y, z);

            Eigen::Vector3d currentCameraPosition = rotation * globalCoord_voxel + translation;
            if (currentCameraPosition.z() <= 0) return;

            Eigen::Vector2i img_coord = SurfaceMeasurement::projectOntoDepthPlane( currentCameraPosition, cameraParams, Cuda::SurfacePrediction::mInf);

            if (!SurfaceMeasurement::containsCuda(img_coord, cameraParams))return;

            const float depth = depthMap[img_coord.y()*depthColumns+ img_coord.x()];
            if (depth <= 0) return;

            auto lambda = calculateLamdas(img_coord, cameraParams);
            auto sdf = calculateSDF(lambda, currentCameraPosition, depth);

            /*
             * SDF Conversion to TSDF & Volumetric Integration
             */
            if (sdf >= -truncationDistance) {

                double current_tsdf =  sdf / truncationDistance; // *sgn(sdf)
                if (current_tsdf < 1) {
                    current_tsdf = 1;
                }
                const double current_weight = 1.0;
                size_t voxel_index = indexX + (indexY * x) + (indexZ * x * y);
                const double old_tsdf = voxels[voxel_index].tsdf;
                const double old_weight = voxels[voxel_index].weight;

                const double updated_tsdf = (old_weight * old_tsdf + current_weight * current_tsdf) /
                    (old_weight + current_weight);
                const double updated_weight = old_weight + current_weight;

                voxels[voxel_index].tsdf = updated_tsdf;
                voxels[voxel_index].weight = updated_weight;

                if (sdf <= truncationDistance / 2 && sdf >= -truncationDistance / 2) {

                    Vector4uc& voxel_color = voxels[voxel_index].color;
                    const Vector4uc image_color = colors[img_coord.x() + (img_coord.y() * cameraParams.imageWidth)];
                    // voxel is invisible
                    if (image_color[3] == 0)
                        return;

                    voxel_color[0] = (old_weight * voxel_color[0] + current_weight * image_color[0]) /
                        (old_weight + current_weight);
                    voxel_color[1] = (old_weight * voxel_color[1] + current_weight * image_color[1]) /
                        (old_weight + current_weight);
                    voxel_color[2] = (old_weight * voxel_color[2] + current_weight * image_color[2]) /
                        (old_weight + current_weight);
                    voxel_color[3] = (old_weight * voxel_color[3] + current_weight * image_color[3]) /
                        (old_weight + current_weight);
                    voxels[voxel_index].color = voxel_color;
                }
            }
        }

    }


    __device__ double calculateLamdas(Eigen::Vector2i& cameraSpacePoint, CameraParametersPyramid cameraParams) {

        const Eigen::Vector3d lambda(
            (cameraSpacePoint.x() - cameraParams.cX) / cameraParams.fovX,
            (cameraSpacePoint.y() - cameraParams.cY) / cameraParams.fovY,
            1.);

        return lambda.norm();
    }

    __device__ double calculateSDF(double& lambda, Eigen::Vector3d& cameraPosition, double rawDepthValue) {
        return (-1.f) * ((1.f / lambda) * cameraPosition.norm() - rawDepthValue);
    }
}