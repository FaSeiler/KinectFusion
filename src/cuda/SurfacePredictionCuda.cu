#include "SurfacePredictionCuda.h"
#include <Eigen/Core>
#include <Eigen/Dense>
#include "Eigen.h"
#include <Eigen/StdVector>
using namespace Eigen;
namespace Cuda {
    namespace Volume {
        __device__ bool intersects(Vector3d* boundsCuda, Ray* r, float entry_distance) {

            float tmin, tmax, tymin, tymax, tzmin, tzmax;

            tmin = (boundsCuda[r->sign[0]].x() - r->origin.x()) * r->invDirection.x();
            tmax = (boundsCuda[1 - r->sign[0]].x() - r->origin.x()) * r->invDirection.x();
            tymin = (boundsCuda[r->sign[1]].y() - r->origin.y()) * r->invDirection.y();
            tymax = (boundsCuda[1 - r->sign[1]].y() - r->origin.y()) * r->invDirection.y();

            if ((tmin > tymax) || (tymin > tmax))
                return false;
            if (tymin > tmin)
                tmin = tymin;
            if (tymax < tmax)
                tmax = tymax;

            tzmin = (boundsCuda[r->sign[2]].z() - r->origin.z()) * r->invDirection.z();
            tzmax = (boundsCuda[1 - r->sign[2]].z() - r->origin.z()) * r->invDirection.z();

            if ((tmin > tzmax) || (tzmin > tmax))
                return false;
            if (tzmin > tmin)
                tmin = tzmin;
            if (tzmax < tmax)
                tmax = tzmax;

            entry_distance = tmin;

            return true;
        }

        __device__ bool contains(Vector3d* _originCuda, Vector3d* _volumeRangeCuda, const Eigen::Vector3d global_point)
        {
            Eigen::Vector3d volumeCoord = (global_point - *_originCuda);

            return !(volumeCoord.x() < 0 || volumeCoord.x() >= _volumeRangeCuda->x() || volumeCoord.y() < 0 ||
                volumeCoord.y() >= _volumeRangeCuda->y() ||
                volumeCoord.z() < 0 || volumeCoord.z() >= _volumeRangeCuda->z());

        }

        __device__ Eigen::Vector3d  getGlobalCoordinate(::Volume* volume, int voxelIdx_x, int voxelIdx_y, int voxelIdx_z){
            const Eigen::Vector3d position((static_cast<double>(voxelIdx_x) + 0.5) * *volume->_voxelScaleCuda,
                (static_cast<double>(voxelIdx_y) + 0.5) * *volume->_voxelScaleCuda,
                (static_cast<double>(voxelIdx_z) + 0.5) * *volume->_voxelScaleCuda);
            return position + *volume->_originCuda;
        }

        __device__ Vector4uc  getColor(::Volume* volume,  Eigen::Vector3d global) {
            Eigen::Vector3d shifted = (global - *volume->_originCuda) / *volume->_voxelScaleCuda;
            Eigen::Vector3i currentPosition;
            currentPosition.x() = int(shifted.x());
            currentPosition.y() = int(shifted.y());
            currentPosition.z() = int(shifted.z());

            Voxel voxelData = volume->_voxelDataCuda[currentPosition.x() + currentPosition.y() * volume->_volumeSizeCuda->x()
                + currentPosition.z() * volume->_volumeSizeCuda->x() *  volume->_volumeSizeCuda->y()];
            return voxelData.color;
        }

        __device__ double  getTSDF(::Volume* volume, Eigen::Vector3d global) {
            Eigen::Vector3d shifted = (global - *volume->_originCuda) / *volume->_voxelScaleCuda;
            Eigen::Vector3i currentPosition;
            currentPosition.x() = int(shifted.x());
            currentPosition.y() = int(shifted.y());
            currentPosition.z() = int(shifted.z());

            Voxel voxelData = volume->_voxelDataCuda[currentPosition.x() + currentPosition.y() * volume->_volumeSizeCuda->x()
                + currentPosition.z() * volume->_volumeSizeCuda->x() * volume->_volumeSizeCuda->y()];
            return voxelData.tsdf;
        }

        __device__ double  getTSDF(::Volume* volume, int x, int y, int z) {
            return
            volume->_voxelDataCuda[x + y * volume->_volumeSizeCuda->x() + z * volume->_volumeSizeCuda->x() * volume->_volumeSizeCuda->y()].tsdf;
        }

    }
    namespace SurfacePrediction {
        __device__  void FillRay(Ray* ray, const Eigen::Vector3d& origin, const Eigen::Vector3d& direction) {
            ray->origin = origin;
            ray->direction = direction;
            //this->invDirection= Eigen::Vector3d(1.0) / this->direction;
            ray->invDirection[0] = (1.0) / ray->direction[0];
            ray->invDirection[1] = (1.0) / ray->direction[1];
            ray->invDirection[2] = (1.0) / ray->direction[2];

            ray->sign[0] = (ray->invDirection.x() < 0);
            ray->sign[1] = (ray->invDirection.y() < 0);
            ray->sign[2] = (ray->invDirection.z() < 0);
        }


        void surfacePrediction(CameraParametersPyramid cameraParams, std::shared_ptr<SurfaceMeasurement> currentFrame, std::shared_ptr<::Volume> volume, float truncationDistance, const int numLevels)
        {

            auto volumeSize = volume->getVolumeSize();
            auto voxelScale = volume->getVoxelScale();
            auto pose = currentFrame->getGlobalPose();
            auto rotationMatrix = pose.block(0, 0, 3, 3);
            auto translation = pose.block(0, 3, 3, 1);
            size_t width = cameraParams.level(numLevels).imageWidth;
            size_t height = cameraParams.level(numLevels).imageHeight;
            const Eigen::Vector3d volumeRange(volumeSize.x() * voxelScale, volumeSize.y() * voxelScale, volumeSize.z() * voxelScale);
             Eigen::Vector3d* volumeRangeCuda;
            cudaMalloc(&volumeRangeCuda, sizeof(Eigen::Vector3d));
            cudaMemcpy(volumeRangeCuda, &volumeRange, sizeof(Eigen::Vector3d), cudaMemcpyHostToDevice);
            
            Eigen::Vector3i* volumeSizeCuda;
             cudaMalloc(&volumeSizeCuda, sizeof(Eigen::Vector3i));
            cudaMemcpy(volumeSizeCuda, &volumeSize, sizeof(Eigen::Vector3i), cudaMemcpyHostToDevice);

                 Eigen::Matrix4d* poseCuda;
             cudaMalloc(&poseCuda, sizeof(Eigen::Matrix4d));
            cudaMemcpy(poseCuda, &pose, sizeof(Eigen::Matrix4d), cudaMemcpyHostToDevice);
            
            Eigen::Block<Eigen::Matrix4d>* rotationCuda;
             cudaMalloc(&rotationCuda, sizeof(Eigen::Block<Eigen::Matrix4d>));
            cudaMemcpy(rotationCuda, &rotationMatrix, sizeof(Eigen::Block<Eigen::Matrix4d>), cudaMemcpyHostToDevice);

            Eigen::Block<Eigen::Matrix4d>* translationCuda;
             cudaMalloc(&translationCuda, sizeof(Eigen::Block<Eigen::Matrix4d>));
            cudaMemcpy(translationCuda, &translation, sizeof(Eigen::Block<Eigen::Matrix4d>), cudaMemcpyHostToDevice);

            CameraParametersPyramid* pyramidCuda;
             cudaMalloc(&pyramidCuda, sizeof(CameraParametersPyramid));
            cudaMemcpy(pyramidCuda, &cameraParams.level(numLevels), sizeof(CameraParametersPyramid), cudaMemcpyHostToDevice);


            // foreach pixel

            // for simplicity, dont optimize the dimensions
            size_t blockSize = 256;
            size_t gridSize = (width * height + 255) / 256;
            size_t vertSize = currentFrame->getGlobalVertexMap().size();
            size_t colorSize = currentFrame->getColorMap().size();
            Eigen::Vector3d* vertices;
            Eigen::Matrix<unsigned char, 4, 1>* colors;
             cudaMalloc(&vertices, vertSize * sizeof(Eigen::Vector3d));
             cudaMalloc(&colors, colorSize * sizeof(Eigen::Matrix<unsigned char, 4, 1>));


           cudaMemcpy(vertices, currentFrame->getGlobalVertexMap().data(), vertSize * sizeof(Eigen::Vector3d), cudaMemcpyHostToDevice);
           cudaMemcpy(colors, currentFrame->getColorMap().data(), colorSize * sizeof(Eigen::Matrix<unsigned char, 4, 1>), cudaMemcpyHostToDevice);
  
            surfacePrediction_kernel<<<gridSize, blockSize>>>(width, height, 
                                                             volumeSizeCuda, volumeRangeCuda, volume->getOriginCuda(), voxelScale,
                                                              poseCuda, rotationCuda, translationCuda, pyramidCuda,
                                                              vertices, colors, truncationDistance, numLevels, volume->boundsCuda);
     
           (const_cast<Eigen::Vector3d*>(currentFrame->getGlobalVertexMap().data()), vertices, vertSize * sizeof(Eigen::Vector3i), cudaMemcpyDeviceToHost);
           (const_cast<Eigen::Matrix<unsigned char, 4, 1>*>(currentFrame->getColorMap().data()), colors, vertSize * sizeof(Eigen::Matrix<unsigned char, 4, 1>), cudaMemcpyDeviceToHost);
            currentFrame->computeGlobalNormalMap(cameraParams.level(numLevels));
            cudaFree(vertices);
            cudaFree(colors);
            cudaFree(volumeRangeCuda);
            cudaFree(volumeSizeCuda);
            cudaFree(poseCuda);
            cudaFree(rotationCuda);
            cudaFree(translationCuda);
            cudaFree(pyramidCuda);

        }

        __global__ void surfacePrediction_kernel(size_t width, size_t height,
            Eigen::Vector3i* volumeSize,  Eigen::Vector3d* volumeRange, Eigen::Vector3d* volumeOrigin, double voxelScale, 
            Eigen::Matrix4d* pose, Eigen::Block<Eigen::Matrix4d>* rotationMatrix, Eigen::Block<Eigen::Matrix4d>* translation,  CameraParametersPyramid* cameraParams,
            Eigen::Vector3d* vertices, Eigen::Matrix<unsigned char, 4, 1>* colors , float truncationDistance, int numLevels, Vector3d* boundsCuda)
        {

            int index = blockIdx.x * blockDim.x + threadIdx.x;

            if (index < width * height) {

                int row = index / width;
                int column = index % width;
                // ray marching in direction
                auto direction = calculateRayDirection(row, column, *rotationMatrix, *cameraParams);

                double rayLength(0.f);

                Ray ray;
                FillRay(&ray, *translation, direction);

                // No intersetcion rule, skip ray
                if (!(Volume::intersects(boundsCuda, &ray, rayLength))) return;
                rayLength += voxelScale;


                Eigen::Vector3d currentPoint = *translation + direction * rayLength;
                if (!Volume::contains(volumeOrigin, volumeRange, currentPoint)) return;

                Eigen::Vector3d previousPoint;

                //double currentTSDF = Volume::getTSDF(volume, currentPoint);
                double previousTSDF;
              
                //for ( double maxLength = rayLength + volumeRange->norm(); rayLength < maxLength; rayLength += truncationDistance) {
                //    previousPoint = currentPoint;
                //    previousTSDF = currentTSDF;

                //    currentPoint = translation + direction * (rayLength + truncationDistance);
                //    if (Volume::contains(volume, currentPoint)) continue;


                //    currentTSDF = Volume::getTSDF(volume, currentPoint);

                //    if (previousTSDF < 0. && currentTSDF > 0.)break;

                //    if (previousTSDF > 0. && currentTSDF < 0.) {
                //        Vector3d globalVertex = getVertexAtZeroCrossing(previousPoint, currentPoint, previousTSDF, currentTSDF);

                //        Vector3d gridVertex = (globalVertex - *volumeOrigin) / voxelScale;

                //        if (!isInBounds(gridVertex, volumeSize)) break;
                //        Vector4uc color;
                //        if (std::abs(previousTSDF) < std::abs(currentTSDF)) {
                //            color = Volume::getColor(volume, previousPoint);
                //        }
                //        else {
                //            color = Volume::getColor(volume, currentPoint);
                //        }

                //        vertices[column * width+ row] = globalVertex;
                //        colors[column* width+ row] = color;

                //    }
                //}
            }
        }


        __device__ Eigen::Vector3d getVertexAtZeroCrossing(
            const Eigen::Vector3d& prevPoint, const Eigen::Vector3d& currPoint,
            double prevTSDF, double currTSDF)
        {
            return (prevPoint * (-currTSDF) + currPoint * prevTSDF) / (prevTSDF - currTSDF);
        }


        __device__  bool isInBounds(const Eigen::Vector3d& gridVertex, Eigen::Vector3i volumeSize) {
            return !(gridVertex.x() - 1 < 1 || gridVertex.x() + 1 >= volumeSize.x() - 1 ||
                gridVertex.y() - 1 < 1 || gridVertex.y() + 1 >= volumeSize.y() - 1 ||
                gridVertex.z() - 1 < 1 || gridVertex.z() + 1 >= volumeSize.z() - 1);
        }


        __device__  Eigen::Vector3d calculateRayDirection(size_t x, size_t y, const Eigen::Matrix3d& rotation,
            CameraParametersPyramid cameraParams) {

            Eigen::Vector3d rayDirection = rotation * Eigen::Vector3d((x - cameraParams.cX) / cameraParams.fovX, (y - cameraParams.cY) / cameraParams.fovY, 1.0);

            rayDirection.normalize();
            return rayDirection;

        }

        __device__ double TSDFInterpolation(const Eigen::Vector3d& gridVertex, ::Volume& volume) {
            Eigen::Vector3i origin = gridVertex.cast<int>();//truncation (floor) as presented in the paper
            Eigen::Vector3d center = origin.cast<double>() + Eigen::Vector3d(0.5f, 0.5f, 0.5f);

            auto deltaX = (gridVertex.x() - center.x());
            auto deltaY = (gridVertex.y() - center.y());
            auto deltaZ = (gridVertex.z() - center.z());

            double accum = 0.0;

            double TSDFS[] = {
                Volume::getTSDF(&volume, (origin.x()), (origin.y()), (origin.z()) * (1 - deltaX) * (1 - deltaY) * (1 - deltaZ)),
                Volume::getTSDF(&volume, (origin.x()), (origin.y()), (origin.z() + 1) * (1 - deltaX) * (1 - deltaY) * (deltaZ)),
                Volume::getTSDF(&volume, (origin.x()), (origin.y() + 1), (origin.z()) * (1 - deltaX) * (deltaY) * (1 - deltaZ)),
                Volume::getTSDF(&volume, (origin.x()), (origin.y() + 1), (origin.z() + 1) * (1 - deltaX) * (deltaY) * (deltaZ)),
                Volume::getTSDF(&volume, (origin.x() + 1), (origin.y()), (origin.z()) * (deltaX) * (1 - deltaY) * (1 - deltaZ)),
                Volume::getTSDF(&volume, (origin.x() + 1), (origin.y()), (origin.z() + 1) * (deltaX) * (1 - deltaY) * (deltaZ)),
                Volume::getTSDF(&volume, (origin.x() + 1), (origin.y() + 1), (origin.z()) * (deltaX) * (deltaY) * (1 - deltaZ)),
                Volume::getTSDF(&volume, (origin.x() + 1), (origin.y() + 1), (origin.z() + 1) * (deltaX) * (deltaY) * (deltaZ))
            };

            for (int i = 0; i < 8; ++i) {
                if (TSDFS[i] == 0) {
                    return mInf;
                }
                accum += TSDFS[i];
            }

            return accum;
        }

        __device__ Eigen::Vector3d calculateNormal(const Eigen::Vector3d& gridVertex, ::Volume& volume) {


            Eigen::Vector3d normal(0.0f, 0.0f, 0.0f);
            Eigen::Vector3d shiftedVertex = gridVertex;

            // x-dir
            shiftedVertex.x()++;
            float tsdf_x1 = TSDFInterpolation(shiftedVertex, volume);
            shiftedVertex = gridVertex;
            shiftedVertex.x()--;

            float tsdf_x2 = TSDFInterpolation(shiftedVertex, volume);
            normal.x() = (tsdf_x1 - tsdf_x2);
            if (tsdf_x1 == mInf || tsdf_x2 == mInf) return Eigen::Vector3d(mInf, mInf, mInf);

            // y-dur
            shiftedVertex = gridVertex;
            shiftedVertex.y()++;
            float tsdf_y1 = TSDFInterpolation(shiftedVertex, volume);
            shiftedVertex = gridVertex;
            shiftedVertex.y()--;

            float tsdf_y2 = TSDFInterpolation(shiftedVertex, volume);
            normal.y() = (tsdf_y1 - tsdf_y2);
            if (tsdf_y1 == mInf || tsdf_y2 == mInf) return Eigen::Vector3d(mInf, mInf, mInf);

            // z-dir

            shiftedVertex = gridVertex;
            shiftedVertex.z()++;
            float tsdf_z1 = TSDFInterpolation(shiftedVertex, volume);
            shiftedVertex = gridVertex;
            shiftedVertex.z()--;

            float tsdf_z2 = TSDFInterpolation(shiftedVertex, volume);
            normal.z() = (tsdf_z1 - tsdf_z2);
            if (tsdf_z1 == mInf || tsdf_z2 == mInf) return Eigen::Vector3d(mInf, mInf, mInf);


            return normal;
        }
    }
}
