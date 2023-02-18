#pragma once

#include <iostream>
#include <fstream>

#include "Eigen.h"
#include "VirtualSensor.h"
#include "SurfaceMeasurement.h"
#include "MarchingCubes.h"

class Mesh
{

public:
	Mesh() {}

    struct Point {
        Point(double x, double y, double z) :_x(x), _y(y), _z(z) {}
        double _x, _y, _z;
        std::string print(std::string color) {
            std::stringstream s;
            s << " " << _x << " " << _y << " " << _z << " " << color << std::endl;
            return s.str();
        }
    };
    struct Square {
        Square(int startIdx, double idx1, double idx2, double idx3, double idx4) :_startIdx(startIdx), _idx1(idx1 + startIdx), _idx2
        (idx2 + startIdx), _idx3(idx3 + startIdx), _idx4(idx4 + startIdx) {

        }

        double _startIdx, _idx1, _idx2, _idx3, _idx4;
        std::string print(std::string color) {
            std::stringstream s;
            s << "4" << " " << _idx1 << " " << _idx2 << " " << _idx4 << " " << _idx3 << " " << color << std::endl;
            return s.str();
        }
    };
    struct Voxel {
        Voxel(double x, double y, double z, double voxelScale, int startIdx) {
            voxelScale *= 0.75;
            vertices.push_back({ x,y,z });
            vertices.push_back({ x + voxelScale,y,z });
            vertices.push_back({ x,y,z + voxelScale });
            vertices.push_back({ x + voxelScale,y,z + voxelScale });
            vertices.push_back({ x,y + voxelScale,z });
            vertices.push_back({ x + voxelScale,y + voxelScale,z });
            vertices.push_back({ x,y + voxelScale,z + voxelScale });
            vertices.push_back({ x + voxelScale,y + voxelScale,z + voxelScale });

            planes.push_back(Square(startIdx, 0, 1, 2, 3));
            planes.push_back(Square(startIdx, 0, 2, 4, 6));
            planes.push_back(Square(startIdx, 0, 1, 4, 5));
            planes.push_back(Square(startIdx, 1, 3, 5, 7));
            planes.push_back(Square(startIdx, 2, 3, 6, 7));
            planes.push_back(Square(startIdx, 4, 5, 6, 7));
        }
        std::string printVertices(std::string color) {
            std::stringstream s;
            for (Point p : vertices)
                s << p.print(color);
            return s.str();
        }
        std::string printPlanes(std::string color) {
            std::stringstream s;
            for (Square p : planes)
                s << " " << p.print(color);
            return s.str();
        }
        std::vector<Point> vertices;
        std::vector<Square> planes;


    };

    static bool writeSurfaceMesh(Mesh mesh, const std::string& filename);
	static bool writeMarchingCubes(std::string filename, Volume& v);
    static bool writeNormalMap(const std::shared_ptr<SurfaceMeasurement>& frame, const std::string& filename);
    static bool writeFileTSDF(std::string filename, Volume& v, int step_size = 1, double threshold = 1, std::string borderColor = "0 255 0");
    static bool writeMeshToFile(std::string filename, const std::shared_ptr<SurfaceMeasurement>& frame, bool includeCameraPose);
	static Mesh joinMeshes(const Mesh& mesh1, const Mesh& mesh2, Matrix4d pose1to2 = Matrix4d::Identity());
	static Mesh camera(const Matrix4d& cameraPose, float scale = 1.f, Vector4uc color = { 255, 0, 0, 255 });

	void clear() {
		m_vertices.clear();
		m_triangles.clear();
	}

	unsigned int addVertex(Vertex& vertex) {
		unsigned int vId = (unsigned int)m_vertices.size();
		m_vertices.push_back(vertex);
		return vId;
	}

	unsigned int addFace(unsigned int idx0, unsigned int idx1, unsigned int idx2) {
		unsigned int fId = (unsigned int)m_triangles.size();
		Triangle triangle(idx0, idx1, idx2);
		m_triangles.push_back(triangle);
		return fId;
	}

	std::vector<Vertex>& getVertices() {
		return m_vertices;
	}

	const std::vector<Vertex>& getVertices() const {
		return m_vertices;
	}

	std::vector<Triangle>& getTriangles() {
		return m_triangles;
	}

	const std::vector<Triangle>& getTriangles() const {
		return m_triangles;
	}

	void transform(const Matrix4d& transformation) {
		for (Vertex& v : m_vertices) {
			v.position = transformation * v.position;
		}
	}

private:
	std::vector<Vertex> m_vertices;
	std::vector<Triangle> m_triangles;

};