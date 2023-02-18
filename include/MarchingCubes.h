#pragma once

#ifndef MARCHING_CUBES_H
#define MARCHING_CUBES_H

#include "MCTables.h"
#include "Volume.h"

#include "Eigen.h"

struct MC_Triangle {
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
		Vector3d p[3];
};

struct MC_Gridcell {
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
		Vector3d p[8];
	double val[8];
};

struct MC_Gridcell_2 {
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
		Vector3d p[8];
	double val[8];
	Vector3d ev0[8];
};

struct VoxelWCoords {
	Voxel _data;
	int _x;
	int _y;
	int _z;
};
struct triangleShape {
	Eigen::Vector3d _idx1;
	Eigen::Vector3d _idx2;
	Eigen::Vector3d _idx3;
	Vector4uc color;

};

class MarchingCubes
{

public:

	static void extractMesh(Volume& volume, std::string fileName);
};

#endif // MARCHING_CUBES_H