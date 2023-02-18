#include "SurfaceMeasurement.h"

SurfaceMeasurement::SurfaceMeasurement(VirtualSensor& sensor, CameraParametersPyramid cameraParams, const float* depthMap, BYTE* colorMap,
    const Eigen::Matrix3d& depthIntrinsics, const Eigen::Matrix3d& colorIntrinsics,
    const Eigen::Matrix4d& d2cExtrinsics,
    const unsigned int width, const unsigned int height, int numLevels, float depthCutoff,
    int kernelSize, float colorSigma,
    float spatialSigma, double maxDistance, unsigned downsampleFactor)
    : m_width(width), m_height(height), m_intrinsic_matrix(depthIntrinsics), m_color_intrinsic_matrix(colorIntrinsics),
    m_d2cExtrinsics(d2cExtrinsics), m_maxDistance(maxDistance)
{
    // init pyramid
    depthPyramid.resize(3);
    smoothedDepthPyramid.resize(3);
    colorPyramid.resize(3);

    vertexPyramid.resize(3);
    normalPyramid.resize(3); 

    depthPyramid[0] = cv::Mat(sensor.getDepthImageHeight(), sensor.getDepthImageWidth(), CV_32FC1, sensor.getDepth());
    colorPyramid[0] = cv::Mat(sensor.getColorImageHeight(), sensor.getColorImageWidth(), CV_8UC4, colorMap);

    // Build pyramids
    for (size_t level = 1; level < num_levels; ++level) {
        cv::resize(depthPyramid[level - 1], depthPyramid[level], cv::Size(sensor.getDepthImageWidth() / 2, sensor.getDepthImageHeight() / 2));
    }

    for (size_t level = 1; level < num_levels; ++level) {
        cv::resize(colorPyramid[level - 1], colorPyramid[level], cv::Size(sensor.getDepthImageWidth() / 2, sensor.getDepthImageHeight() / 2));
    }

    // filter bilateral
    for (size_t level = 0; level < num_levels; ++level) {
        cv::bilateralFilter(depthPyramid[level], // source
            smoothedDepthPyramid[level], // destination
            kernelSize,
            colorSigma,
            spatialSigma,
            cv::BORDER_DEFAULT);
    }

    setDepthMapPyramid(smoothedDepthPyramid[0]);
    setColorMapPyramid(colorPyramid[0]);

    std::vector<Eigen::Vector3d> rawVertexMap = computeVertexMap(smoothedDepthPyramid[0], cameraParams.level(numLevels));

    std::vector<Eigen::Vector3d> rawNormalMap = computeNormalMap(rawVertexMap, cameraParams.level(numLevels), maxDistance);
    filterMask(rawVertexMap, rawNormalMap, downsampleFactor);
    setGlobalPose(Eigen::Matrix4d::Identity());

    computeColorMap(colorPyramid[0], cameraParams.level(numLevels), numLevels);

    //------------------------------------------ OpenCV CUDA begin ------------------------------------------//
#if OPENCV_CUDA

    // init pyramid
    depthPyramidCuda.resize(3);
    smoothedDepthPyramidCuda.resize(3);
    colorPyramidCuda.resize(3);

    vertexPyramidCuda.resize(3);
    normalPyramidCuda.resize(3);

    allocateMemory(cameraParams);

    // Start by uploading original frame to GPU
    depthPyramidCuda[0].upload(cv::Mat(sensor.getDepthImageHeight(), sensor.getDepthImageWidth(), CV_16UC1, sensor.getDepth()));

    // Build pyramids and filter bilaterally on GPU
    cv::cuda::Stream stream;
    for (size_t level = 1; level < num_levels; ++level)
        cv::cuda::pyrDown(depthPyramidCuda[level - 1], depthPyramidCuda[level], stream);
    for (size_t level = 0; level < num_levels; ++level) {
        cv::cuda::bilateralFilter(depthPyramidCuda[level], // source
            smoothedDepthPyramidCuda[level], // destination
            kernelSize,
            colorSigma,
            spatialSigma,
            cv::BORDER_DEFAULT,
            stream);
    }
    stream.waitForCompletion();

    // Compute vertex and normal maps
    for (size_t level = 0; level < num_levels; ++level) {
        computeVertexMapCuda(smoothedDepthPyramidCuda[level], vertexPyramidCuda[level],
            depthCutoff, cameraParams.level(level));
        computeNormalMapCuda(vertexPyramidCuda[level], normalPyramidCuda[level]);
    }

    colorPyramidCuda[0].upload(cv::Mat(sensor.getColorImageHeight(), sensor.getColorImageWidth(), CV_8UC4, colorMap));

#endif
    //------------------------------------------ OpenCV CUDA end ------------------------------------------//
}

//------------------------------------------ OpenCV CUDA begin ------------------------------------------//

#if OPENCV_CUDA

__global__ void kernelComputeVertexMap(const PtrStepSz<float> depthMap, PtrStep<double3> vertexMap,
    const float depthCutoff, CameraParametersPyramid cameraParams)
{

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= depthMap.cols || y >= depthMap.rows)
        return;

    float depthValue = depthMap.ptr(y)[x];
    if (depthValue > depthCutoff) depthValue = 0.f;

    Vec3da vertex;

    if (depthValue == mInf) {
        vertex = Vec3da(mInf, mInf, mInf);

        vertexMap.ptr(y)[x] = make_double3(vertex.x(), vertex.y(), vertex.z());
    }
    else {
        vertex = Vec3da(
            (x - cameraParams.cX) / cameraParams.fovX * depthValue,
            (y - cameraParams.cY) / cameraParams.fovY * depthValue,
            depthValue);

        vertexMap.ptr(y)[x] = make_double3(vertex.x(), vertex.y(), vertex.z());
    }
}

void SurfaceMeasurement::computeVertexMapCuda(const GpuMat& depthMap, GpuMat& vertexMap, const float depthCutoff,
    CameraParametersPyramid cameraParams)
{
    dim3 threads(32, 32);
    dim3 blocks((depthMap.cols + threads.x - 1) / threads.x, (depthMap.rows + threads.y - 1) / threads.y);

    kernelComputeVertexMap << < blocks, threads >> > (depthMap, vertexMap, depthCutoff, cameraParams);

    cudaThreadSynchronize();
}

__global__ void kernelComputeNormalMap(const PtrStepSz<double3> vertexMap, PtrStep<double3> normalMap)
{

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < 1 || x >= vertexMap.cols - 1 || y < 1 || y >= vertexMap.rows - 1)
        return;

    const Vec3da left(&vertexMap.ptr(y)[x - 1].x);
    const Vec3da right(&vertexMap.ptr(y)[x + 1].x);
    const Vec3da upper(&vertexMap.ptr(y - 1)[x].x);
    const Vec3da lower(&vertexMap.ptr(y + 1)[x].x);

    Vec3da normal;

    if (left.z() == 0 || right.z() == 0 || upper.z() == 0 || lower.z() == 0)
        normal = Vec3da(0.f, 0.f, 0.f);
    else {
        Vec3da du(left.x() - right.x(), left.y() - right.y(), left.z() - right.z());
        Vec3da dv(upper.x() - lower.x(), upper.y() - lower.y(), upper.z() - lower.z());

        normal = du.cross(dv);
        normal.normalize();

        if (normal.z() > 0)
            normal *= -1;
    }

    normalMap.ptr(y)[x] = make_double3(normal.x(), normal.y(), normal.z());
}

void SurfaceMeasurement::computeNormalMapCuda(const GpuMat& vertexMap, GpuMat& normalMap)
{
    dim3 threads(32, 32);
    dim3 blocks((vertexMap.cols + threads.x - 1) / threads.x,
        (vertexMap.rows + threads.y - 1) / threads.y);

    kernelComputeNormalMap << < blocks, threads >> > (vertexMap, normalMap);

    cudaThreadSynchronize();
}

void SurfaceMeasurement::allocateMemory(CameraParametersPyramid cameraParams) {

    // Cuda
    // Allocate GPU memory
    for (size_t level = 0; level < num_levels; ++level) {
        const int width = cameraParams.level(level).imageWidth;
        const int height = cameraParams.level(level).imageHeight;

        depthPyramidCuda[level] = cv::cuda::createContinuous(height, width, CV_32FC1);
        smoothedDepthPyramidCuda[level] = cv::cuda::createContinuous(height, width, CV_32FC1);
        colorPyramidCuda[level] = cv::cuda::createContinuous(height, width, CV_8UC4);

        vertexPyramidCuda[level] = cv::cuda::createContinuous(height, width, CV_32FC3);
        normalPyramidCuda[level] = cv::cuda::createContinuous(height, width, CV_32FC3);
    }
}

__global__ void kernelComputeTranformMaps(int rows, int cols, const PtrStep<float> vertexMap, const PtrStep<float> normalMap,
    const Eigen::Matrix3f& rotation, const float3 translation, PtrStepSz<float> globalVertexMapDst, PtrStep<float> globalNormalMapDst)
{
    int x =  blockIdx.x * blockDim.x + threadIdx.x;
    int y =  blockIdx.y * blockDim.y + threadIdx.y;

    if (x < cols && y < rows)
    {
        //vertices
        float3 globalVertexSrc = make_float3(mInf, mInf, mInf);
        globalVertexSrc.x = vertexMap.ptr(y)[x];

        Eigen::Vector3f globalVertexDst;

        Eigen::Vector3f translationVec(translation.x, translation.y, translation.z);

        if (!isnan(globalVertexSrc.x))
        {
            globalVertexSrc.y = vertexMap.ptr(y + rows)[x];
            globalVertexSrc.z = vertexMap.ptr(y + 2 * rows)[x];

            Eigen::Vector3f globalVertexVec(globalVertexSrc.x, globalVertexSrc.y, globalVertexSrc.z);

            globalVertexDst = rotation * globalVertexVec + translationVec;

            globalVertexMapDst.ptr(y + rows)[x] = globalVertexDst[1];
            globalVertexMapDst.ptr(y + 2 * rows)[x] = globalVertexDst[2];
        }

        globalVertexMapDst.ptr(y)[x] = globalVertexDst[0];

        //normals
        float3 globalNormalSrc = make_float3(mInf, mInf, mInf);
        globalNormalSrc.x = normalMap.ptr(y)[x];

        Eigen::Vector3f globalNormalDst;

        if (!isnan(globalNormalSrc.x))
        {
            globalNormalSrc.y = normalMap.ptr(y + rows)[x];
            globalNormalSrc.z = normalMap.ptr(y + 2 * rows)[x];

            Eigen::Vector3f globalNormalVec(globalNormalSrc.x, globalNormalSrc.y, globalNormalSrc.z);

            globalNormalDst = rotation * globalNormalVec;

            globalNormalMapDst.ptr(y + rows)[x] = globalNormalDst[1];
            globalNormalMapDst.ptr(y + 2 * rows)[x] = globalNormalDst[2];
        }

        globalNormalMapDst.ptr(y)[x] = globalNormalDst[0];
    }
}
  
void SurfaceMeasurement::computeTranformMapsCuda(const GpuMat& vertexMap, const GpuMat& normalMap,
    const Eigen::Matrix3f& rotation, const float3& translation,
    GpuMat& globalVertexMapDst, GpuMat& globalNormalMapDst)
{
    int cols = vertexMap.cols;
    int rows = vertexMap.rows / 3;

    globalVertexMapDst.create(rows * 3, cols, CV_32FC3);
    globalNormalMapDst.create(rows * 3, cols, CV_32FC3);

    dim3 threads(32, 32);
    dim3 blocks((vertexMap.cols + threads.x - 1) / threads.x,
        (vertexMap.rows + threads.y - 1) / threads.y);

    kernelComputeTranformMaps << <blocks, threads >> > (rows, cols, vertexMap, normalMap, rotation, translation, globalVertexMapDst, globalNormalMapDst);
    
    cudaThreadSynchronize();
}

void SurfaceMeasurement::setDepthMapPyramidCuda(const GpuMat& depthMap) {
    depthMap.assignTo(m_depth_map_pyramid_cuda);
}

const GpuMat& SurfaceMeasurement::getDepthMapPyramidCuda() const {
    return m_depth_map_pyramid_cuda;
}

void SurfaceMeasurement::setColorMapPyramidCuda(const GpuMat& colorMap) {
    colorMap.assignTo(m_color_map_pyramid_cuda);
}

const GpuMat& SurfaceMeasurement::getColorMapPyramidCuda() const {
    return m_color_map_pyramid_cuda ;
}

#endif

//------------------------------------------ OpenCV CUDA end ------------------------------------------//

std::vector<Eigen::Vector3d> SurfaceMeasurement::computeVertexMap(cv::Mat& depthMapConvert, CameraParametersPyramid cameraParams) {

    // Back-project the pixel depths into the camera space.
    std::vector<Vector3d> rawVertexMap(cameraParams.imageWidth * cameraParams.imageHeight);

    for (int v = 0; v < cameraParams.imageHeight; ++v) {
        // For every pixel in a row.
        for (int u = 0; u < cameraParams.imageWidth; ++u) {
            unsigned int idx = v * cameraParams.imageWidth + u; // linearized index
            float depth = depthMapConvert.at<float>(v, u);
            if (depth == MINF) {
                rawVertexMap[idx] = Vector3d(MINF, MINF, MINF);
            }
            else {
                // Back-projection to camera space.
                rawVertexMap[idx] = Vector3d((u - cameraParams.cX) / cameraParams.fovX * depth, (v - cameraParams.cY) / cameraParams.fovY * depth, depth);
            }
        }
    }

    return rawVertexMap;
}

std::vector<Eigen::Vector3d> SurfaceMeasurement::computeNormalMap(std::vector<Eigen::Vector3d> vertexMap, CameraParametersPyramid cameraParams, double maxDistance) {

    // We need to compute derivatives and then the normalized normal vector (for valid pixels).
    std::vector<Eigen::Vector3d> rawNormalMap(cameraParams.imageWidth * cameraParams.imageHeight);
    const float maxDistanceHalved = maxDistance / 2.f;

    for (int v = 1; v < cameraParams.imageHeight - 1; ++v) {
        for (int u = 1; u < cameraParams.imageWidth - 1; ++u) {
            unsigned int idx = v * cameraParams.imageWidth + u; // linearized index

            const Eigen::Vector3d du = vertexMap[idx + 1] - vertexMap[idx - 1];
            const Eigen::Vector3d dv = vertexMap[idx + cameraParams.imageWidth] - vertexMap[idx - cameraParams.imageWidth];
            if (!du.allFinite() || !dv.allFinite() || du.norm() > maxDistanceHalved || dv.norm() > maxDistanceHalved) {
                rawNormalMap[idx] = Eigen::Vector3d(MINF, MINF, MINF);
                continue;
            }

            // Compute the normals using central differences. 
            rawNormalMap[idx] = du.cross(dv);
            rawNormalMap[idx].normalize();
        }
    }

    // We set invalid normals for border regions.
    for (int u = 0; u < cameraParams.imageWidth; ++u) {
        rawNormalMap[u] = Eigen::Vector3d(MINF, MINF, MINF);
        rawNormalMap[u + (cameraParams.imageHeight - 1) * cameraParams.imageWidth] = Eigen::Vector3d(MINF, MINF, MINF);
    }
    for (int v = 0; v < cameraParams.imageHeight; ++v) {
        rawNormalMap[v * cameraParams.imageWidth] = Eigen::Vector3d(MINF, MINF, MINF);
        rawNormalMap[(cameraParams.imageWidth - 1) + v * cameraParams.imageWidth] = Eigen::Vector3d(MINF, MINF, MINF);
    }

    return rawNormalMap;
}

void SurfaceMeasurement::filterMask(std::vector<Eigen::Vector3d> rawVertexMap, std::vector<Eigen::Vector3d> rawNormalMap, int downsampleFactor)
{
    // We filter out measurements where either point or normal is invalid.
    const unsigned nVertices = rawVertexMap.size();
    vertexMap.reserve(std::floor(float(nVertices) / downsampleFactor));
    normalMap.reserve(std::floor(float(nVertices) / downsampleFactor));

    for (int i = 0; i < nVertices; i = i + downsampleFactor) {
        const auto& vertex = rawVertexMap[i];
        const auto& normal = rawNormalMap[i];

        if ((vertex.allFinite() && normal.allFinite())) {
            vertexMap.push_back(vertex);
            normalMap.push_back(normal);
        }
        else {
            vertexMap.emplace_back(Eigen::Vector3d(MINF, MINF, MINF));
            normalMap.emplace_back(Eigen::Vector3d(MINF, MINF, MINF));
        }
    }
}

void SurfaceMeasurement::computeColorMap(cv::Mat& colorMap, CameraParametersPyramid cameraParams, int numLevels) {
    const auto rotation = m_d2cExtrinsics.block(0, 0, 3, 3);
    const auto translation = m_d2cExtrinsics.block(0, 3, 3, 1);

    std::vector<Vector4uc> colors(cameraParams.imageWidth * cameraParams.imageHeight);
    for (int v = 0; v < cameraParams.imageHeight; ++v) {
        // For every pixel in a row.
        for (int u = 0; u < cameraParams.imageWidth; ++u) {
            unsigned int idx = v * cameraParams.imageWidth + u; // linearized index
            colors[idx] = Vector4uc(colorMap.at<cv::Vec4b>(v, u)[0], colorMap.at<cv::Vec4b>(v, u)[1], colorMap.at<cv::Vec4b>(v, u)[2], colorMap.at<cv::Vec4b>(v, u)[3]);
        }
    }
    BYTE zero = (BYTE)255;

    m_color_map.reserve(vertexMap.size());
    for (size_t i = 0; i < vertexMap.size(); ++i) {
        Eigen::Vector2i coord = projectOntoColorPlane(rotation * vertexMap[i] + translation, cameraParams.level(numLevels));
        if (contains(coord, cameraParams.level(numLevels)))
            m_color_map.push_back(colors[coord.x() + coord.y() * cameraParams.imageWidth]);
        else
            m_color_map.push_back(Vector4uc(zero, zero, zero, zero));
    }
}

void SurfaceMeasurement::computeTriangles(std::vector<Eigen::Vector3d> rawVertexMap, CameraParametersPyramid cameraParams, float edgeThreshold) {
    
    Matrix4d depthExtrinsicsInv = m_d2cExtrinsics.inverse();
    
    std::vector<Eigen::Vector3d> globalRawVertexMap = transformPoints(rawVertexMap, depthExtrinsicsInv);

    // Compute inverse camera pose (mapping from camera CS to world CS).
    //Matrix4d cameraPoseInverse = cameraPose.inverse();
    
    triangles.reserve((cameraParams.imageHeight - 1) * (cameraParams.imageWidth - 1) * 2);

    // Compute triangles (faces).
    for (unsigned int i = 0; i < cameraParams.imageHeight - 1; i++) {
        for (unsigned int j = 0; j < cameraParams.imageWidth - 1; j++) {
            unsigned int i0 = i * cameraParams.imageWidth + j;
            unsigned int i1 = (i + 1) * cameraParams.imageWidth + j;
            unsigned int i2 = i * cameraParams.imageWidth + j + 1;
            unsigned int i3 = (i + 1) * cameraParams.imageWidth + j + 1;

            bool valid0 = globalRawVertexMap[i0].allFinite();
            bool valid1 = globalRawVertexMap[i1].allFinite();
            bool valid2 = globalRawVertexMap[i2].allFinite();
            bool valid3 = globalRawVertexMap[i3].allFinite();

            if (valid0 && valid1 && valid2) {
                float d0 = (globalRawVertexMap[i0] - globalRawVertexMap[i1]).norm();
                float d1 = (globalRawVertexMap[i0] - globalRawVertexMap[i2]).norm();
                float d2 = (globalRawVertexMap[i1] - globalRawVertexMap[i2]).norm();
                if (edgeThreshold > d0 && edgeThreshold > d1 && edgeThreshold > d2)
                    addFace(i0, i1, i2);
            }
            if (valid1 && valid2 && valid3) {
                float d0 = (globalRawVertexMap[i3] - globalRawVertexMap[i1]).norm();
                float d1 = (globalRawVertexMap[i3] - globalRawVertexMap[i2]).norm();
                float d2 = (globalRawVertexMap[i1] - globalRawVertexMap[i2]).norm();
                if (edgeThreshold > d0 && edgeThreshold > d1 && edgeThreshold > d2)
                    addFace(i1, i3, i2);
            }
        }
    }
}

Eigen::Matrix3d SurfaceMeasurement::intrinsicPyramid(CameraParametersPyramid cameraParams) {

    Eigen::Matrix3d intrinsics;
    intrinsics << cameraParams.fovX, 0.0f, cameraParams.cX, 0.0f, cameraParams.fovY, cameraParams.cY, 0.0f, 0.0f, 1.0f;

    return intrinsics;
}


Eigen::Vector3d SurfaceMeasurement::projectIntoCamera(const Eigen::Vector3d& globalCoord) {
    Eigen::Matrix4d pose_inverse = m_global_pose.inverse();
    const auto rotation_inv = pose_inverse.block(0, 0, 3, 3);
    const auto translation_inv = pose_inverse.block(0, 3, 3, 1);
    return rotation_inv * globalCoord + translation_inv;
}

bool SurfaceMeasurement::contains(const Eigen::Vector2i& img_coord, CameraParametersPyramid cameraParams) {
    return img_coord[0] < cameraParams.imageWidth && img_coord[1] < cameraParams.imageHeight && img_coord[0] >= 0 && img_coord[1] >= 0;
}

Eigen::Vector2i SurfaceMeasurement::projectOntoPlane(const Eigen::Vector3d& cameraCoord, Eigen::Matrix3d& intrinsics) {
    Eigen::Vector3d projected = (intrinsics * cameraCoord);
    if (projected[2] == 0) {
        return Eigen::Vector2i(MINF, MINF);
    }
    projected /= projected[2];
    return (Eigen::Vector2i((int)round(projected.x()), (int)round(projected.y())));
}
Eigen::Vector2i SurfaceMeasurement::projectOntoDepthPlane(const Eigen::Vector3d& cameraCoord, CameraParametersPyramid cameraParams) {
    m_intrinsic_matrix = intrinsicPyramid(cameraParams);

    return projectOntoPlane(cameraCoord, m_intrinsic_matrix);
}


Eigen::Vector2i SurfaceMeasurement::projectOntoColorPlane(const Eigen::Vector3d& cameraCoord, CameraParametersPyramid cameraParams) {
    m_color_intrinsic_matrix = intrinsicPyramid(cameraParams);

    return projectOntoPlane(cameraCoord, m_color_intrinsic_matrix);
}

void SurfaceMeasurement::computeGlobalNormalMap(CameraParametersPyramid cameraParams) {
    std::vector<Eigen::Vector3d> camera_points;
    for (const auto& global : globalVertexMap) {
        camera_points.emplace_back(projectIntoCamera(global));
    }
    globalNormalMap = computeNormalMap(camera_points, cameraParams, m_maxDistance);
}

void SurfaceMeasurement::applyGlobalPose(Eigen::Matrix4d& estimated_pose) {
    Eigen::Matrix3d rotation = estimated_pose.block(0, 0, 3, 3);

    globalVertexMap = transformPoints(vertexMap, estimated_pose);
    globalNormalMap = rotatePoints(normalMap, rotation);
}

std::vector<Eigen::Vector3d> SurfaceMeasurement::transformPoints(std::vector<Eigen::Vector3d>& points, Eigen::Matrix4d& transformation) {
    const Eigen::Matrix3d rotation = transformation.block(0, 0, 3, 3);
    const Eigen::Vector3d translation = transformation.block(0, 3, 3, 1);
    std::vector<Eigen::Vector3d> transformed(points.size());

    for (size_t idx = 0; idx < points.size(); ++idx) {
        if (points[idx].allFinite())
            transformed[idx] = rotation * points[idx] + translation;
        else
            transformed[idx] = (Eigen::Vector3d(MINF, MINF, MINF));
    }
    return transformed;
}

std::vector<Eigen::Vector3d> SurfaceMeasurement::rotatePoints(std::vector<Eigen::Vector3d>& points, Eigen::Matrix3d& rotation) {
    std::vector<Eigen::Vector3d> transformed(points.size());

    for (size_t idx = 0; idx < points.size(); ++idx) {
        if (points[idx].allFinite())
            transformed[idx] = rotation * points[idx];
        else
            transformed[idx] = (Eigen::Vector3d(MINF, MINF, MINF));
    }
    return transformed;
}

const std::vector<Eigen::Vector3d>& SurfaceMeasurement::getVertexMap() const {
    return vertexMap;
}

const std::vector<Eigen::Vector3d>& SurfaceMeasurement::getNormalMap() const {
    return normalMap;
}

const std::vector<Eigen::Vector3d>& SurfaceMeasurement::getGlobalNormalMap() const {
    return globalNormalMap;
}

void SurfaceMeasurement::setGlobalNormalMap(const Eigen::Vector3d& normal, size_t u, size_t v, CameraParametersPyramid cameraParams) {
    size_t idx = v * cameraParams.imageWidth + u;
    globalNormalMap[idx] = normal;
}

const std::vector<Eigen::Vector3d>& SurfaceMeasurement::getGlobalVertexMap() const {
    return globalVertexMap;
}

void SurfaceMeasurement::setGlobalVertexMap(const Eigen::Vector3d& point, size_t u, size_t v, CameraParametersPyramid cameraParams) {
    size_t idx = v * cameraParams.imageWidth + u;
    globalVertexMap[idx] = point;
}

const Eigen::Matrix4d& SurfaceMeasurement::getGlobalPose() const {
    return m_global_pose;
}

void SurfaceMeasurement::setGlobalPose(const Eigen::Matrix4d& pose) {
    m_global_pose = pose;
    applyGlobalPose(m_global_pose);
}

const std::vector<Vector4uc>& SurfaceMeasurement::getColorMap() const {
    return m_color_map;
}

void SurfaceMeasurement::setColor(const Vector4uc& color, size_t u, size_t v, CameraParametersPyramid cameraParams) {
    size_t idx = v * cameraParams.imageWidth + u;
    m_color_map[idx] = color;
}

void SurfaceMeasurement::setDepthMapPyramid(const cv::Mat& depthMap) {
    m_depth_map_pyramid = depthMap;
}

const cv::Mat& SurfaceMeasurement::getDepthMapPyramid() const {
    return m_depth_map_pyramid;
}

void SurfaceMeasurement::setColorMapPyramid(const cv::Mat& colorMap) {
    m_color_map_pyramid = colorMap;
}

const cv::Mat& SurfaceMeasurement::getColorMapPyramid() const {
    return m_color_map_pyramid;
}