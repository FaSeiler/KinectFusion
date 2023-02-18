#pragma once

#include "Volume.h"
#include <SurfaceMeasurement.h>
#include <memory>
#include "VirtualSensor.h"

namespace SurfaceReconstruction {
    //THIS method expects frame to hold all camera paramerters as well as the estimated pose --> TODO: check if those values are set or redefine method parameters

     void reconstructSurface(CameraParametersPyramid cameraParams, std::shared_ptr<SurfaceMeasurement>& currentFrame, const std::shared_ptr<Volume>& volume, double truncationDistance, int numLevels);

    /*!
         * The original implementation actually takes a raw depth Value, as we already calculated the camereSpacePoints
         * only the normalization has to be done.
         * TODO: move normalization to frame.cpp ; check if the cameraSpaceTransformation in frame.cpp equals the one used in Paper
         * @param cameraSpacePoint
         * @return the normalized cameraSpaceCoordinates
         */
    double calculateLamdas(Eigen::Vector2i& cameraSpacePoint, CameraParametersPyramid cameraParams);
    /*!
     *
     * @param lambda
     * @param cameraPosition
     * @param rawDepthValue
     * @return the signed-distance-function for the specific depth value lambda is based on
     */

    double calculateSDF(double& lambda, Eigen::Vector3d& cameraPosition, double rawDepthValue);

};