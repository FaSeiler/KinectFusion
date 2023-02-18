// KinectFusion.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <fstream>

#include "Eigen.h"
#include "VirtualSensor.h"
#include "Mesh.h"
#include "SurfaceMeasurement.h"
#include "SurfacePoseEstimation.h"
#include "Volume.h"
#include "SurfacePrediction.h"
#include "SurfacePredictionCuda.h"
#include "SurfaceReconstructionCuda.h"
#include "SurfaceReconstruction.h"
#include "Ray.h"
#include "SurfacePoseEstimationCuda.h"

#define WRITE_MESH 0
#define WRITE_NORMAL_MAP 0
#define WRITE_MESH_WITH_CAMERA 0
#define WRITE_MARCHING_CUBES 0
#define WRITE_TSDF 0

#define DATASET_FREIBURG1_XYZ 0
#define DATASET_FREIBURG2_XYZ 1

VirtualSensor sensor;

bool process_frame(SurfacePoseEstimation* optimizer, CameraParametersPyramid cameraParams, size_t frame_cnt, std::shared_ptr<SurfaceMeasurement> firstFrame, std::shared_ptr<SurfaceMeasurement> currentFrame, 
    std::shared_ptr<Volume> volume, const Config& config)
{
    Eigen::Matrix4d estimated_pose = firstFrame->getGlobalPose();
    currentFrame->setGlobalPose(estimated_pose);

    // STEP 2: Surface Pose Estimation
    std::cout << "Init: Surface Pose Estimation..." << std::endl;
    clock_t begin_icp = clock();
    optimizer->EstimateTransform(currentFrame, firstFrame, estimated_pose);
    clock_t end_icp = clock();
    double deltaTime_icp = double(end_icp - begin_icp) / CLOCKS_PER_SEC;

    std::cout << "Surface Pose Estimation: " << deltaTime_icp << " seconds." << std::endl;

    if (USE_CUDA) {
        // STEP 3: Surface Reconstruction
        std::cout << "Init: Fusion Cuda..." << std::endl;
        volume->CopyToCuda();

        SurfaceReconstructionCuda::reconstructSurface(cameraParams, currentFrame, volume, config.m_truncationDistance, config.m_num_levels);
        volume->CopyFromCuda();

        // STEP 4: Surface Prediction
        std::cout << "Init: Raycast Cuda..." << std::endl;
        Cuda::SurfacePrediction::surfacePrediction(cameraParams, currentFrame, volume, (float)config.m_truncationDistance, config.m_num_levels);
    }
    else {
        // STEP 3: Surface Reconstruction
        std::cout << "Init: Surface Reconstruction..." << std::endl;
        clock_t begin_fusion = clock();
        SurfaceReconstruction::reconstructSurface(cameraParams, currentFrame, volume, config.m_truncationDistance, config.m_num_levels);
        clock_t end_fusion = clock();
        double deltaTime_fusion = double(end_fusion - begin_fusion) / CLOCKS_PER_SEC;

        std::cout << "Surface Reconstruction: " << deltaTime_fusion << " seconds." << std::endl;

        // STEP 4: Surface Prediction
        std::cout << "Init: Surface Prediction..." << std::endl;
        clock_t begin_raycast = clock();
        SurfacePrediction::surfacePrediction(cameraParams, currentFrame, volume, config.m_truncationDistance, config.m_num_levels);
        clock_t end_raycast = clock();
        double deltaTime_raycast = double(end_raycast - begin_raycast) / CLOCKS_PER_SEC;

        std::cout << "Surface Prediction: " << deltaTime_raycast << " seconds." << std::endl;
    }
    std::cout << "Done!" << std::endl;

    return true;
}

int main() {

    std::string filenameIn;

    if (DATASET_FREIBURG1_XYZ) {
        filenameIn = std::string("../../Data/rgbd_dataset_freiburg1_xyz/");
    }
    else if (DATASET_FREIBURG2_XYZ) {
        filenameIn = std::string("../../Data/rgbd_dataset_freiburg2_xyz/");
    }
    else {
        filenameIn = std::string("../../Data/rgbd_dataset_freiburg1_xyz/");
    }

    std::cout << "Initialize virtual sensor..." << std::endl;

    if (!sensor.init(filenameIn)) {
        std::cout << "Failed to initialize the sensor!\nCheck file path!" << std::endl;
        return -1;
    }

    /*
     * Performance Measure
     */

    int total_number_frames = 0;
    double deltaTime_pipeline_total = 0.0;

    /*
     * Configuration Stuff
     */
    Eigen::Vector3d volumeRange(5.0, 5.0, 5.0);
    Eigen::Vector3i volumeSize(512, 512, 512);
    double voxelSize = volumeRange.x() / volumeSize.x();

    const auto volumeOrigin = Eigen::Vector3d(-volumeRange.x() / 2, -volumeRange.y() / 2, 0.5);

    Config config(0.1, 0.5, 0.06, volumeOrigin, volumeSize.x(), volumeSize.y(), volumeSize.z(), voxelSize, 1500.0f);

    //print Configuration to File
    config.printToFile("config");

    /*
     * Setting up the Volume from Configuration
     */
    auto volume = std::make_shared<Volume>(config.m_volumeOrigin, config.m_volumeSize, config.m_voxelScale);

    /*
     * Process a first frame as a reference frame.
     * --> All next frames are tracked relatively to the first frame.
     */
    sensor.processNextFrame();
    Eigen::Matrix3d depthIntrinsics = sensor.getDepthIntrinsics();
    Eigen::Matrix3d colIntrinsics = sensor.getColorIntrinsics();
    Eigen::Matrix4d d2cExtrinsics = sensor.getDepthExtrinsics();
    const unsigned int depthWidth = sensor.getDepthImageWidth();
    const unsigned int depthHeight = sensor.getDepthImageHeight();

    CameraParametersPyramid cameraParams{};
    cameraParams.imageWidth = sensor.getDepthImageWidth();
    cameraParams.imageHeight = sensor.getDepthImageHeight();
    cameraParams.fovX = depthIntrinsics(0, 0);
    cameraParams.fovY = depthIntrinsics(1, 1);
    cameraParams.cX = depthIntrinsics(0, 2);
    cameraParams.cY = depthIntrinsics(1, 2);

    const float* depthMap = &sensor.getDepth()[0];
    BYTE* colors = &sensor.getColorRGBX()[0];

    // Optimize using ICP
    SurfacePoseEstimation* optimizer = new SurfacePoseEstimation(); //rename

    optimizer->SetKNNDistance(0.1f);
    optimizer->SetNrIterations(10);
    optimizer->LogCeresUpdates(false, false);

    std::shared_ptr<SurfaceMeasurement> firstFrame = std::make_shared<SurfaceMeasurement>(SurfaceMeasurement(sensor, cameraParams, depthMap, colors, depthIntrinsics, colIntrinsics, 
        d2cExtrinsics, depthWidth, depthHeight, config.m_num_levels, config.m_depth_cutoff));
    
    if (WRITE_MESH)
    {
        if (WRITE_MESH_WITH_CAMERA)
        {
            Mesh::writeMeshToFile("frame0.off", firstFrame, true);
        }
        else
        {
            Mesh::writeMeshToFile("frame0.off", firstFrame, false);
        }
    }

    int i = 1;
    const int iMax = 50;

    while (i <= iMax && sensor.processNextFrame()) {

        clock_t begin_pipeline = clock();

        const float* depthMap = &sensor.getDepth()[0];
        BYTE* colors = &sensor.getColorRGBX()[0];
        // STEP 1: Surface Measurement
        std::cout << "Init: Surface Measurement..." << std::endl;
        clock_t begin_surface_measurement = clock();
        std::shared_ptr<SurfaceMeasurement> currentFrame = std::make_shared<SurfaceMeasurement>(SurfaceMeasurement(sensor, cameraParams, depthMap, colors, depthIntrinsics, colIntrinsics, 
            d2cExtrinsics, depthWidth, depthHeight, config.m_num_levels, config.m_depth_cutoff));
        clock_t end_surface_measurement = clock();
        double deltaTime_surface_measurement = double(end_surface_measurement - begin_surface_measurement) / CLOCKS_PER_SEC;

        std::cout << "Surface Measurement: " << deltaTime_surface_measurement << " seconds." << std::endl;
        
        process_frame(optimizer, cameraParams, i, firstFrame, currentFrame, volume, config);

        if ((i - 1) % 5 == 0) {
            // Write Mesh
            if (WRITE_MESH)
            {
                std::stringstream filename_mesh;
                filename_mesh << "frame" << i << ".off";

                if (WRITE_NORMAL_MAP) {
                    Mesh::writeNormalMap(currentFrame, std::string("normalmap_") + std::to_string(i));
                }

                if (WRITE_MESH_WITH_CAMERA)
                {
                    Mesh::writeMeshToFile(filename_mesh.str(), currentFrame, true);
                }
                else
                {
                    Mesh::writeMeshToFile(filename_mesh.str(), currentFrame, false);
                }
            }

            //Write Fused Volume to File with Marching Cubes Algorithm
            if (WRITE_MARCHING_CUBES)
            {
                Mesh::writeMarchingCubes(std::string("marchingCubes_") + std::to_string(i), *volume);
            }

            //Write Fused Volume to File with Blocks indicating the Distance of each Voxel
            if (WRITE_TSDF)
            {
                Mesh::writeFileTSDF(std::string("tsdf_") + std::to_string(i), *volume);
            }
        }

        i++;

        clock_t end_pipeline = clock();
        double deltaTime_pipeline = double(end_pipeline - begin_pipeline) / CLOCKS_PER_SEC;

        std::cout << "Pipeline " << deltaTime_pipeline << " seconds." << std::endl;

        total_number_frames += 1;
        deltaTime_pipeline_total += deltaTime_pipeline;

        std::cout << "Pipeline Frames: " << total_number_frames << " " << std::endl;
        std::cout << "Pipeline Total time: " << deltaTime_pipeline_total << " seconds." << std::endl;
    }
}

