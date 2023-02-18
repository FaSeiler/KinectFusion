#include <iostream>
#include <SurfaceReconstruction.h>
namespace SurfaceReconstruction {
    void reconstructSurface(CameraParametersPyramid cameraParams, std::shared_ptr<SurfaceMeasurement>& currentFrame, const std::shared_ptr<Volume>& volume, double truncationDistance, int numLevels) {

        auto volumeSize = volume->getVolumeSize();
        auto pose = currentFrame->getGlobalPose().inverse();
        auto voxelData = volume->getVoxelData();
        int idx = 0;
        cv::Mat depthMap = currentFrame->getDepthMapPyramid();

        Eigen::Matrix3d rotation = pose.block(0, 0, 3, 3);
        Eigen::Vector3d translation = pose.block(0, 3, 3, 1);

        for (int z = 0; z < volumeSize.z(); z++) {
            for (int y = 0; y < volumeSize.y(); y++) {
                for (int x = 0; x < volumeSize.x(); x++) {
                    /*
                     * Volumetric Reconstruction
                     */
                     //calculate Camera Position
                    Eigen::Vector3d globalCoord_voxel = volume->getGlobalCoordinate(x, y, z);
                    Eigen::Vector3d currentCameraPosition = rotation * globalCoord_voxel + translation;
                    if (currentCameraPosition.z() <= 0) continue;

                    Eigen::Vector2i img_coord = currentFrame->projectOntoDepthPlane(currentCameraPosition, cameraParams.level(numLevels));

                    if (!currentFrame->contains(img_coord, cameraParams.level(numLevels)))continue;

                    const float depth = depthMap.at<float>(img_coord.y(), img_coord.x());
                    if (depth <= 0) continue;

                    auto lambda = calculateLamdas(img_coord, cameraParams.level(numLevels));
                    auto sdf = calculateSDF(lambda, currentCameraPosition, depth);

                    /*
                     * SDF Conversion to TSDF & Volumetric Integration
                     */
                    if (sdf >= -truncationDistance) {
                        idx++;

                        const double current_tsdf = std::min(1., sdf / truncationDistance); // *sgn(sdf)
                        const double current_weight = 1.0;
                        size_t voxel_index = x + (y * volumeSize.x()) + (z * volumeSize.x() * volumeSize.y());
                        const double old_tsdf = volume->getVoxelData()[voxel_index].tsdf;
                        const double old_weight = volume->getVoxelData()[voxel_index].weight;

                        const double updated_tsdf = (old_weight * old_tsdf + current_weight * current_tsdf) /
                            (old_weight + current_weight);
                        const double updated_weight = old_weight + current_weight;

                        volume->getVoxelData()[voxel_index].tsdf = updated_tsdf;
                        volume->getVoxelData()[voxel_index].weight = updated_weight;

                        if (sdf <= truncationDistance / 2 && sdf >= -truncationDistance / 2) {

                            Vector4uc& voxel_color = voxelData[voxel_index].color;
                            const Vector4uc image_color = currentFrame->getColorMap()[img_coord.x() + (img_coord.y() * cameraParams.level(numLevels).imageWidth)];
                            // voxel is invisible
                            if (image_color[3] == 0)
                                continue;

                            voxel_color[0] = (old_weight * voxel_color[0] + current_weight * image_color[0]) /
                                (old_weight + current_weight);
                            voxel_color[1] = (old_weight * voxel_color[1] + current_weight * image_color[1]) /
                                (old_weight + current_weight);
                            voxel_color[2] = (old_weight * voxel_color[2] + current_weight * image_color[2]) /
                                (old_weight + current_weight);
                            voxel_color[3] = (old_weight * voxel_color[3] + current_weight * image_color[3]) /
                                (old_weight + current_weight);
                            volume->getVoxelData()[voxel_index].color = voxel_color;
                        }
                    }
                }
            }
        }
    }

    double calculateLamdas(Eigen::Vector2i& cameraSpacePoint, CameraParametersPyramid cameraParams) {

        const Eigen::Vector3d lambda(
            (cameraSpacePoint.x() - cameraParams.cX) / cameraParams.fovX,
            (cameraSpacePoint.y() - cameraParams.cY) / cameraParams.fovY,
            1.);

        return lambda.norm();
    }

    double calculateSDF(double& lambda, Eigen::Vector3d& cameraPosition, double rawDepthValue) {
        return (-1.f) * ((1.f / lambda) * cameraPosition.norm() - rawDepthValue);
    }
}