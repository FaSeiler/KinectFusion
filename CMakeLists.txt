cmake_minimum_required(VERSION 3.20)

set(CMAKE_CONFIGURATION_TYPES Debug Release CACHE TYPE INTERNAL FORCE)

project(KinectFusion LANGUAGES C CXX CUDA)

# Options
set(LIBRARY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../Libs CACHE PATH "Path to lib folder")
set(Eigen3_DIR ${LIBRARY_DIR}/Eigen/share/eigen3/cmake CACHE PATH "Path to installed Eigen")
set(glog_DIR ${LIBRARY_DIR}/glog-lib/lib/cmake/glog CACHE PATH "Path to installed glog")
set(Ceres_DIR ${LIBRARY_DIR}/Ceres/CMake CACHE PATH "Path to installed Ceres")
set(Flann_INCLUDE_DIR ${LIBRARY_DIR}/Flann-1.8.4/ CACHE PATH "Path to Flann source folder")
#set(OpenCV_INCLUDE_DIRS ${LIBRARY_DIR}/opencv-3.4.14/cmake CACHE PATH "Path to OpenCV source folder")

# ------------------------------------------------
#                 Dependencies
# ------------------------------------------------

# CUDA
find_package(CUDAToolkit REQUIRED)

# OpenCV
#find_package(OpenCV REQUIRED)
SET("OpenCV_DIR" "C:/KinectFusion/Development/Libs/opencv-4.7.0/Build")
find_package(OpenCV 4.0 REQUIRED)
if (OpenCV_INCLUDE_DIRS)
    include_directories("${OpenCV_INCLUDE_DIRS}")
    link_directories(${OpenCV_DIR}/lib)
endif (OpenCV_INCLUDE_DIRS)

# Use pre-compiled libraries on Windows
if(WIN32)
    if(CMAKE_SIZEOF_VOID_P EQUAL 8)
        set(FreeImage_INCLUDE_DIR ${LIBRARY_DIR}/FreeImage-3.18.0/Dist/x64/ CACHE PATH "Path to FreeImage header file")
        set(FreeImage_LIBRARY_DIR ${LIBRARY_DIR}/FreeImage-3.18.0/Dist/x64/ CACHE PATH "Path to FreeImage .lib/.dll folder")
    elseif(CMAKE_SIZEOF_VOID_P EQUAL 4)
        set(FreeImage_INCLUDE_DIR ${LIBRARY_DIR}/FreeImage-3.18.0/Dist/x32/ CACHE PATH "Path to FreeImage header file")
        set(FreeImage_LIBRARY_DIR ${LIBRARY_DIR}/FreeImage-3.18.0/Dist/x32/ CACHE PATH "Path to FreeImage .lib/.dll folder")
    endif()
endif(WIN32)

find_package(Ceres REQUIRED)
find_package(glog REQUIRED)
get_target_property(GLOG_DLL_PATH_DEBUG glog::glog IMPORTED_LOCATION_DEBUG)
get_target_property(GLOG_DLL_PATH_RELEASE glog::glog IMPORTED_LOCATION_RELEASE)

add_definitions("-D_DISABLE_EXTENDED_ALIGNED_STORAGE")

# ------------------------------------------------

include_directories(include)

set(HEADER_FILES 
    include/DataTypes.h
    include/Eigen.h
    include/VirtualSensor.h
    include/FreeImageHelper.h
    include/SurfaceMeasurement.h
    include/MarchingCubes.h
    include/MCTables.h
    include/Volume.h
    include/Mesh.h
    include/SurfacePoseEstimation.h
    include/NearestNeighbor.h
    include/SurfacePrediction.h
    include/Ray.h
    include/SurfaceReconstruction.h
    include/SurfaceReconstructionCuda.h
    include/SurfacePredictionCuda.h
    include/SurfacePoseEstimationCuda.h
    include/SurfacePredictionCuda.h
)

set(SOURCE_FILES 
    src/FreeImageHelper.cpp
    src/MarchingCubes.cpp
    src/VirtualSensor.cpp
    src/Mesh.cpp
    src/SurfacePoseEstimation.cpp
    src/SurfacePrediction.cpp
    src/SurfaceReconstruction.cpp
    src/cuda/SurfaceMeasurement.cu
    src/cuda/SurfaceReconstructionCuda.cu
    src/cuda/SurfacePoseEstimationCuda.cu
    src/cuda/SurfacePredictionCuda.cu
    src/cuda/SurfaceReconstructionCuda.cu
    src/cuda/Volume.cu
)

link_directories(${FreeImage_LIBRARY_DIR})

add_executable(KinectFusion src/main.cpp ${HEADER_FILES} ${SOURCE_FILES})

set_target_properties(KinectFusion
   PROPERTIES
       CUDA_SEPARABLE_COMPILATION ON
       CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

#target_link_libraries(KinectFusion ceres freeimage Eigen3::Eigen CUDA::cudart ${OpenCV_LIBS})
target_link_libraries(KinectFusion ceres freeimage Eigen3::Eigen ${OpenCV_LIBS})
target_include_directories(KinectFusion PUBLIC ${EIGEN3_INCLUDE_DIR} ${FreeImage_INCLUDE_DIR} ${Flann_INCLUDE_DIR} ${OpenCV_INCLUDE_DIRS})

# On Windows copy dll to executable folder
if(WIN32)
    # Visual Studio properties
    set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT KinectFusion)
    set_property(TARGET KinectFusion PROPERTY VS_DEBUGGER_WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/)

    # On Windows copy dll to executable folder
    add_custom_command(TARGET KinectFusion POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${FreeImage_LIBRARY_DIR}/FreeImage.dll"
            ${CMAKE_BINARY_DIR})
    
    add_custom_command(TARGET KinectFusion POST_BUILD        
        COMMAND ${CMAKE_COMMAND} -E copy_if_different          
            "${GLOG_DLL_PATH_DEBUG}"                           
            ${CMAKE_BINARY_DIR})                     
    
    add_custom_command(TARGET KinectFusion POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${GLOG_DLL_PATH_RELEASE}"
            ${CMAKE_BINARY_DIR})
endif(WIN32)
