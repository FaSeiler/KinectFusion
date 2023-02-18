#pragma once
#include "Eigen.h"
class Ray
{
public:
    Eigen::Vector3d origin, direction, invDirection;
    int sign[3];
};
