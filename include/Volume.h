#pragma once
#include "cuda_runtime.h"
#include "Eigen.h"
#include "DataTypes.h"
#include "Ray.h"
#include <limits>
typedef unsigned int uint;
#define USE_CUDA 0
class Volume {
public:
    Volume(const Eigen::Vector3d origin, const Eigen::Vector3i volumeSize, const double voxelScale);
    ~Volume();

    bool intersects(const Ray& r, float& entry_distance) const;

    std::vector<Voxel>& getVoxelData();

    const Eigen::Vector3d& getOrigin() const;

    const Eigen::Vector3i& getVolumeSize() const;
    float getVoxelScale() const;

    bool contains(const Eigen::Vector3d point);

    Eigen::Vector3d Volume::getGlobalCoordinate(int voxelIdx_x, int voxelIdx_y, int voxelIdx_z);

    void CopyToCuda();
    void CopyFromCuda();

    double getTSDF(Eigen::Vector3d global);
    double getTSDF(int x, int y, int z);


    Vector4uc getColor(Eigen::Vector3d global);
    Eigen::Vector3d getTSDFGrad(Eigen::Vector3d global);
    const Eigen::Vector3d getOrigin();
    Eigen::Vector3d* getOriginCuda();
    const double getVoxelScale();


private:
    //_voxelData contains color, tsdf & Weight
    std::vector<Voxel> _voxelData;
    const Eigen::Vector3i _volumeSize;
    const double _voxelScale;
    const Eigen::Vector3d _volumeRange;

    const Eigen::Vector3d _origin;
    const Eigen::Vector3d _maxPoint;
    Eigen::Vector3d bounds[2];
public:
    //_voxelData contains color, tsdf & Weight
    Voxel* _voxelDataCuda;
    Eigen::Vector3i* _volumeSizeCuda;
    double* _voxelScaleCuda;
    Eigen::Vector3d* _volumeRangeCuda;

    Eigen::Vector3d* _originCuda;
    Eigen::Vector3d* _maxPointCuda;
    Eigen::Vector3d* boundsCuda;

};


