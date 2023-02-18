
#pragma once
#include <algorithm>
#include <fstream>

#include <limits>
#include <cmath>

#include <vector>
#include "DataTypes.h"
#include "Eigen.h"
#include "VirtualSensor.h"
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#define OPENCV_CUDA 0

#ifndef MINF
#define MINF -std::numeric_limits<double>::infinity()
#endif

//------------------------------------------ OpenCV CUDA begin ------------------------------------------//
#if OPENCV_CUDA 

#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>

#include <opencv2/core/cuda.hpp>

using cv::cuda::GpuMat;
using cv::cuda::PtrStep;
using cv::cuda::PtrStepSz;

using Vec3fda = Eigen::Matrix<float, 3, 1, Eigen::DontAlign>;
using Vec3da = Eigen::Matrix<double, 3, 1, Eigen::DontAlign>;

#endif
//------------------------------------------ OpenCV CUDA end ------------------------------------------//

struct Vertex {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    // Position stored as 4 floats (4th component is supposed to be 1.0)
    Vector4d position;
    // Color stored as 4 unsigned char
    Vector4uc color;
};

struct Triangle {
    unsigned int idx0;
    unsigned int idx1;
    unsigned int idx2;

    Triangle() : idx0{ 0 }, idx1{ 0 }, idx2{ 0 } {}

    Triangle(unsigned int _idx0, unsigned int _idx1, unsigned int _idx2) :
        idx0(_idx0), idx1(_idx1), idx2(_idx2) {}
};

struct CameraParametersPyramid {
    int imageWidth, imageHeight;
    float fovX, fovY;
    float cX, cY;

    /**
     * Returns camera parameters for a specified pyramid level; each level corresponds to a scaling of pow(.5, level)
     * @param level The pyramid level to get the parameters for with 0 being the non-scaled version,
     * higher levels correspond to smaller spatial size
     * @return A CameraParameters structure containing the scaled values
     */
    CameraParametersPyramid level(const size_t level) const
    {
        if (level == 0) return *this;

        const float scale_factor = powf(0.5f, static_cast<float>(level));
        return CameraParametersPyramid{ imageWidth >> level, imageHeight >> level,
                                  fovX * scale_factor, fovY * scale_factor,
                                  (cX + 0.5f) * scale_factor - 0.5f,
                                  (cY + 0.5f) * scale_factor - 0.5f };
    }
};

__device__ static float mInf = -std::numeric_limits<float>::infinity();

class SurfaceMeasurement {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        SurfaceMeasurement(VirtualSensor& sensor, CameraParametersPyramid cameraParams, const float* depthMap, BYTE* colorMap, const Eigen::Matrix3d& depthIntrinsics, const Eigen::Matrix3d& colorIntrinsics,
            const Eigen::Matrix4d& d2cExtrinsics,
            const unsigned int width, const unsigned int height, int num_levels, float depthCutoff, int bfilter_kernel_size = 5, float bfilter_color_sigma = 1.f,
            float bfilter_spatial_sigma = 1.f, double maxDistance = 2, unsigned downsampleFactor = 1);

    //------------------------------------------ OpenCV CUDA begin ------------------------------------------//
#if OPENCV_CUDA 

    void computeVertexMapCuda(const GpuMat& depthMap, GpuMat& vertexMap, const float depthCutoff,
        CameraParametersPyramid cameraParams);
    void computeNormalMapCuda(const GpuMat& vertexMap, GpuMat& normalMap);

    void allocateMemory(CameraParametersPyramid cameraParams);

    void setDepthMapPyramidCuda(const GpuMat& depthMap);

    const GpuMat& getDepthMapPyramidCuda() const;

    void setColorMapPyramidCuda(const GpuMat& colorMap);

    const GpuMat& getColorMapPyramidCuda() const;

    void computeTranformMapsCuda(const GpuMat& vertexMap, const GpuMat& normalMap,
        const Eigen::Matrix3f& rotation, const float3& translation,
        GpuMat& globalVertexMapDst, GpuMat& globalNormalMapDst);

#endif
    //------------------------------------------ OpenCV CUDA end ------------------------------------------//

    void applyGlobalPose(Eigen::Matrix4d& estimated_pose);

    const std::vector<Eigen::Vector3d>& getVertexMap() const;

    const std::vector<Eigen::Vector3d>& getNormalMap() const;

    const std::vector<Eigen::Vector3d>& getGlobalVertexMap() const;

    void setGlobalVertexMap(const Eigen::Vector3d& point, size_t u, size_t v, CameraParametersPyramid cameraParams);

    const std::vector<Eigen::Vector3d>& getGlobalNormalMap() const;

    void setGlobalNormalMap(const Eigen::Vector3d& normal, size_t u, size_t v, CameraParametersPyramid cameraParams);

    void computeGlobalNormalMap(CameraParametersPyramid cameraParams);

    const Eigen::Matrix4d& getGlobalPose() const;

    void setGlobalPose(const Eigen::Matrix4d& pose);

    void setDepthMapPyramid(const cv::Mat& depthMap);

    const cv::Mat& getDepthMapPyramid() const;

    void setColorMapPyramid(const cv::Mat& colorMap);

    const cv::Mat& getColorMapPyramid() const;

    const std::vector<Vector4uc>& getColorMap() const;

    void setColor(const Vector4uc& color, size_t u, size_t v, CameraParametersPyramid cameraParams);

    bool contains(const Eigen::Vector2i& point, CameraParametersPyramid cameraParams);

    Eigen::Vector3d projectIntoCamera(const Eigen::Vector3d& globalCoord);

    Eigen::Vector2i projectOntoDepthPlane(const Eigen::Vector3d& cameraCoord, CameraParametersPyramid cameraParams);
    Eigen::Vector2i projectOntoColorPlane(const Eigen::Vector3d& cameraCoord, CameraParametersPyramid cameraParams);

    Eigen::Matrix3d intrinsicPyramid(CameraParametersPyramid cameraParams);
    unsigned int addFace(unsigned int idx0, unsigned int idx1, unsigned int idx2) {
        unsigned int fId = (unsigned int)triangles.size();
        Triangle triangle(idx0, idx1, idx2);
        triangles.push_back(triangle);
        return fId;
    }

private:
    Eigen::Vector2i projectOntoPlane(const Eigen::Vector3d& cameraCoord, Eigen::Matrix3d& intrinsics);

    void computeColorMap(cv::Mat& colorMap, CameraParametersPyramid cameraParams, int numLevels);

    void computeTriangles(std::vector<Eigen::Vector3d> rawVertexMap, CameraParametersPyramid cameraParams, float edgeThreshold);

    std::vector<Eigen::Vector3d> computeVertexMap(cv::Mat& depthMapConvert, CameraParametersPyramid cameraParams);

    std::vector<Eigen::Vector3d> computeNormalMap(std::vector<Eigen::Vector3d> vertexMap, CameraParametersPyramid cameraParams, double maxDistance);

    void filterMask(std::vector<Eigen::Vector3d> rawVertexMap, std::vector<Eigen::Vector3d> rawNormalMap, int downsampleFactor);
    std::vector<Eigen::Vector3d> transformPoints(std::vector<Eigen::Vector3d>& points, Eigen::Matrix4d& transformation);

    std::vector<Eigen::Vector3d> rotatePoints(std::vector<Eigen::Vector3d>& points, Eigen::Matrix3d& rotation);


    std::vector<Eigen::Vector3d> vertexMap;
    std::vector<Eigen::Vector3d> normalMap;

    const unsigned int m_width;
    const unsigned int m_height;

    std::vector<Eigen::Vector3d> globalVertexMap;
    std::vector<Eigen::Vector3d> globalNormalMap;
    Eigen::Matrix4d m_global_pose;
public: Eigen::Matrix3d m_intrinsic_matrix;
private:
    Eigen::Matrix3d m_color_intrinsic_matrix;
    Eigen::Matrix4d m_d2cExtrinsics;

    cv::Mat smoothedDepthMapConvert;

    cv::Mat m_depth_map_pyramid;
    cv::Mat m_color_map_pyramid;

    std::vector<double> m_depth_map;
    std::vector<Vector4uc> m_color_map;

    int bfilter_kernel_size{ 5 };
    float bfilter_color_sigma{ 1.f };
    float bfilter_spatial_sigma{ 1.f };

    // The number of pyramid levels to generate for each frame, including the original frame level
    int num_levels{ 3 };

    double m_maxDistance;

    std::vector<Triangle> triangles;

    std::vector<cv::Mat> depthPyramid;
    std::vector<cv::Mat> smoothedDepthPyramid;
    std::vector<cv::Mat> colorPyramid;

    std::vector<cv::Mat> vertexPyramid;
    std::vector<cv::Mat> normalPyramid;

    //------------------------------------------ OpenCV CUDA begin ------------------------------------------//
#if OPENCV_CUDA 
    std::vector<GpuMat> depthPyramidCuda;
    std::vector<GpuMat> smoothedDepthPyramidCuda;
    std::vector<GpuMat> colorPyramidCuda;

    std::vector<GpuMat> vertexPyramidCuda;
    std::vector<GpuMat> normalPyramidCuda;

    GpuMat globalVertexPyramidCuda;
    GpuMat globalNormalPyramidCuda;

    GpuMat m_depth_map_pyramid_cuda;
    GpuMat m_color_map_pyramid_cuda;

#endif
    //------------------------------------------ OpenCV CUDA end ------------------------------------------//
};
