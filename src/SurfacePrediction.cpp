#include "SurfacePrediction.h"
#include <Eigen/Core>
#include <Eigen/Dense>
#include "Eigen.h"
#include <Eigen/StdVector>

using namespace Eigen;

void SurfacePrediction::FillRay(Ray* ray, const Eigen::Vector3d& origin, const Eigen::Vector3d& direction) {
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

bool calculatePointOnRay(Eigen::Vector3d& currentPoint,
    std::shared_ptr<Volume>& volume,
    const Eigen::Vector3d& origin,
    const Eigen::Vector3d& direction,
    float raylength
) {
    currentPoint = (origin + (direction * raylength));
    return volume->contains(currentPoint);
}

void SurfacePrediction::surfacePrediction(CameraParametersPyramid cameraParams, std::shared_ptr<SurfaceMeasurement>& currentFrame, std::shared_ptr<Volume>& volume, float truncationDistance, int numLevels)
{
    //params that i should get from someone else somehow
    auto volumeSize = volume->getVolumeSize();
    auto voxelScale = volume->getVoxelScale();
    auto pose = currentFrame->getGlobalPose();
    auto rotationMatrix = pose.block(0, 0, 3, 3);
    auto translation = pose.block(0, 3, 3, 1);
    auto width = cameraParams.level(numLevels).imageWidth;
    auto height = cameraParams.level(numLevels).imageHeight;

    std::vector<double> depthMap(width * height);

    const Eigen::Vector3d volumeRange(volumeSize.x() * voxelScale, volumeSize.y() * voxelScale, volumeSize.z() * voxelScale);

    // foreach pixel
    for (auto row = 0; row < height; ++row) {
        for (auto column = 0; column < width; ++column) {
            // ray marching in direction
            auto direction = calculateRayDirection(column, row, rotationMatrix, cameraParams.level(numLevels));

            float rayLength(0.f);

            Ray ray;
            FillRay(&ray, translation, direction);
            // No intersetcion rule, skip ray
            if (!(volume->intersects(ray, rayLength))) continue;

            rayLength += voxelScale;
            Eigen::Vector3d currentPoint = translation + direction * rayLength;
            if (!volume->contains(currentPoint)) continue;

            Eigen::Vector3d previousPoint;


            double currentTSDF = volume->getTSDF(currentPoint);
            double previousTSDF;


            for (double maxLength = rayLength + volumeRange.norm(); rayLength < maxLength; rayLength += truncationDistance) {
                previousPoint = currentPoint;
                previousTSDF = currentTSDF;

                currentPoint = translation + direction * (rayLength + truncationDistance);
                if (!volume->contains(currentPoint)) continue;

                currentTSDF = volume->getTSDF(currentPoint);

                if (previousTSDF < 0. && currentTSDF > 0.)break;

                if (previousTSDF > 0. && currentTSDF < 0.) {
                    Vector3d globalVertex = getVertexAtZeroCrossing(previousPoint, currentPoint, previousTSDF, currentTSDF);

                    Vector3d gridVertex = (globalVertex - volume->getOrigin()) / voxelScale;

                    if (!isInBounds(gridVertex, volumeSize)) break;
                    Vector4uc color;
                    if (std::abs(previousTSDF) < std::abs(currentTSDF)) {
                        color = volume->getColor(previousPoint);
                    }
                    else {
                        color = volume->getColor(currentPoint);
                    }

                    currentFrame->setGlobalVertexMap(globalVertex, column, row, cameraParams.level(numLevels));
                    currentFrame->setColor(color, column, row, cameraParams.level(numLevels));
                }
            }
        }
    }

    currentFrame->computeGlobalNormalMap(cameraParams.level(numLevels));
}

bool  SurfacePrediction::isInBounds(const Eigen::Vector3d& gridVertex, Eigen::Vector3i volumeSize) {
    return !(gridVertex.x() - 1 < 1 || gridVertex.x() + 1 >= volumeSize.x() - 1 ||
        gridVertex.y() - 1 < 1 || gridVertex.y() + 1 >= volumeSize.y() - 1 ||
        gridVertex.z() - 1 < 1 || gridVertex.z() + 1 >= volumeSize.z() - 1);
}

Eigen::Vector3d SurfacePrediction::calculateRayDirection(size_t x, size_t y, const Eigen::Matrix3d& rotation, CameraParametersPyramid cameraParams) {


    Eigen::Vector3d rayDirection = rotation * Eigen::Vector3d((x - cameraParams.cX) / cameraParams.fovX, (y - cameraParams.cY) / cameraParams.fovY, 1.0);

    rayDirection.normalize();
    return rayDirection;

}

Eigen::Vector3d SurfacePrediction::getVertexAtZeroCrossing(
    const Eigen::Vector3d& prevPoint, const Eigen::Vector3d& currPoint,
    const double prevTSDF, const double currTSDF)
{
    return (prevPoint * (-currTSDF) + currPoint * prevTSDF) / (prevTSDF - currTSDF);
}

double SurfacePrediction::TSDFInterpolation(const Eigen::Vector3d& gridVertex,
    const std::shared_ptr<Volume>& volume) {
    Eigen::Vector3i origin = gridVertex.cast<int>();//truncation, maybe check if ddifferent rule applies
    Eigen::Vector3d center = origin.cast<double>() + Eigen::Vector3d(0.5f, 0.5f, 0.5f);

    auto deltaX = (gridVertex.x() - center.x());
    auto deltaY = (gridVertex.y() - center.y());
    auto deltaZ = (gridVertex.z() - center.z());

    double accum = 0.0;

    double TSDFS[] = {
        volume->getTSDF((origin.x()), (origin.y()), (origin.z()) * (1 - deltaX) * (1 - deltaY) * (1 - deltaZ)),
        volume->getTSDF((origin.x()), (origin.y()), (origin.z() + 1) * (1 - deltaX) * (1 - deltaY) * (deltaZ)),
        volume->getTSDF((origin.x()), (origin.y() + 1), (origin.z()) * (1 - deltaX) * (deltaY) * (1 - deltaZ)),
        volume->getTSDF((origin.x()), (origin.y() + 1), (origin.z() + 1) * (1 - deltaX) * (deltaY) * (deltaZ)),
        volume->getTSDF((origin.x() + 1), (origin.y()), (origin.z()) * (deltaX) * (1 - deltaY) * (1 - deltaZ)),
        volume->getTSDF((origin.x() + 1), (origin.y()), (origin.z() + 1) * (deltaX) * (1 - deltaY) * (deltaZ)),
        volume->getTSDF((origin.x() + 1), (origin.y() + 1), (origin.z()) * (deltaX) * (deltaY) * (1 - deltaZ)),
        volume->getTSDF((origin.x() + 1), (origin.y() + 1), (origin.z() + 1) * (deltaX) * (deltaY) * (deltaZ))
    };

    for (int i = 0; i < 8; ++i) {
        if (TSDFS[i] == 0) {
            return MINF;
        }
        accum += TSDFS[i];
    }

    return accum;
}


Eigen::Vector3d SurfacePrediction::calculateNormal(const Eigen::Vector3d& gridVertex,
    const std::shared_ptr<Volume>& volume, float truncationDistance) {

    Eigen::Vector3d normal(0.0f, 0.0f, 0.0f);
    Eigen::Vector3d shiftedVertex = gridVertex;

    // x-dir
    shiftedVertex.x()++;
    float tsdf_x1 = TSDFInterpolation(shiftedVertex, volume);
    shiftedVertex = gridVertex;
    shiftedVertex.x()--;

    float tsdf_x2 = TSDFInterpolation(shiftedVertex, volume);
    normal.x() = (tsdf_x1 - tsdf_x2);
    if (tsdf_x1 == MINF || tsdf_x2 == MINF) return Eigen::Vector3d(MINF, MINF, MINF);

    // y-dur
    shiftedVertex = gridVertex;
    shiftedVertex.y()++;
    float tsdf_y1 = TSDFInterpolation(shiftedVertex, volume);
    shiftedVertex = gridVertex;
    shiftedVertex.y()--;

    float tsdf_y2 = TSDFInterpolation(shiftedVertex, volume);
    normal.y() = (tsdf_y1 - tsdf_y2);
    if (tsdf_y1 == MINF || tsdf_y2 == MINF) return Eigen::Vector3d(MINF, MINF, MINF);

    // z-dir

    shiftedVertex = gridVertex;
    shiftedVertex.z()++;
    float tsdf_z1 = TSDFInterpolation(shiftedVertex, volume);
    shiftedVertex = gridVertex;
    shiftedVertex.z()--;

    float tsdf_z2 = TSDFInterpolation(shiftedVertex, volume);
    normal.z() = (tsdf_z1 - tsdf_z2);
    if (tsdf_z1 == MINF || tsdf_z2 == MINF) return Eigen::Vector3d(MINF, MINF, MINF);

    return normal;
}