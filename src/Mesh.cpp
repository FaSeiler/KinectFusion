#include "Mesh.h"

bool Mesh::writeSurfaceMesh(Mesh mesh, const std::string& filename) {
	
	std::cout << "Writing " << filename << std::endl;

	// Write off file.
	std::string filenameBaseOut = std::string("../result/mesh_");
	std::ofstream outFile(filenameBaseOut + filename + ".off");
	if (!outFile.is_open()) return false;
	
	// Write header.
	outFile << "COFF" << std::endl;
	outFile << mesh.getVertices().size() << " " << mesh.getTriangles().size() << " 0" << std::endl;
	
	// Save vertices.
	for (unsigned int i = 0; i < mesh.getVertices().size(); i++) {
		const auto& vertex = mesh.getVertices()[i];
		if (vertex.position.allFinite())
		{
			outFile << vertex.position.x() << " " << vertex.position.y() << " " << vertex.position.z() << " "
				<< int(vertex.color.x()) << " " << int(vertex.color.y()) << " " << int(vertex.color.z()) << " " << int(vertex.color.w()) << std::endl;
		}
		else {
			outFile << "0.0 0.0 0.0 0 0 0 0" << std::endl;
		}
	}

	// Save faces.
	for (unsigned int i = 0; i < mesh.getTriangles().size(); i++) {
		outFile << "3 " << mesh.getTriangles()[i].idx0 << " " << mesh.getTriangles()[i].idx1 << " " << mesh.getTriangles()[i].idx2 << std::endl;
	}

	// Close file.
	outFile.close();

	return true;
}

bool Mesh::writeMarchingCubes(std::string filename, Volume& v) {
	std::cout << "Writing " << filename << std::endl;
	MarchingCubes::extractMesh(v, filename);
	return true;
}

bool Mesh::writeFileTSDF(std::string filename, Volume& v, int step_size, double threshold, std::string borderColor) {

	std::cout << "Writing " << filename << std::endl;

	std::string filenameBaseOut = std::string("../result/mesh_");

	// Write off file.
	std::ofstream outFile(filenameBaseOut + filename + ".off");
	if (!outFile.is_open()) {
		return false;
	}
	auto volumeSize = v.getVolumeSize();
	// Save vertices.
	std::vector<Voxel> voxels;
	std::vector<std::string> colors;

	Eigen::Vector3d red(255, 0, 0);
	Eigen::Vector3d blue(0, 0, 255);

	int idx = 0;
	for (int z = 0; z < volumeSize.z(); z += step_size) {
		for (int y = 0; y < volumeSize.y(); y += step_size) {
			for (int x = 0; x < volumeSize.x(); x += step_size) {
				auto voxel = v.getVoxelData()[x + y * volumeSize.x() + z * volumeSize.x() * volumeSize.y()];
				if (voxel.weight == 0. || std::abs(voxel.tsdf) >= threshold) {
					continue;
				}
				voxels.push_back(Voxel(v.getOrigin().x() + x * v.getVoxelScale(), v.getOrigin().y() + y * v.getVoxelScale
				(), v.getOrigin().z() + z * v.getVoxelScale(), v.getVoxelScale() * step_size, idx));

				// max value for TSDF 1, min value -1
				double tsdf_abs = std::abs(voxel.tsdf);
				Eigen::Vector3i col = ((1 - tsdf_abs) * red + tsdf_abs * blue).cast<int>();
				std::stringstream s;
				s << col.x() << " " << col.y() << " " << col.z();
				colors.push_back(s.str());
				idx += 8;
			}
		}
	}

	std::vector<Voxel> borders;
	for (int x = 0; x < volumeSize.x(); ++x) {
		Eigen::Vector3d p1 = (v.getOrigin() + Eigen::Vector3d(x, 0, 0) * v.getVoxelScale());
		Eigen::Vector3d p2 = (v.getOrigin() + Eigen::Vector3d(x, volumeSize.y(), 0) * v.getVoxelScale());
		Eigen::Vector3d p3 = (v.getOrigin() + Eigen::Vector3d(x, volumeSize.y(), volumeSize.z()) * v.getVoxelScale());
		Eigen::Vector3d p4 = (v.getOrigin() + Eigen::Vector3d(x, 0, volumeSize.z()) * v.getVoxelScale());
		borders.push_back(Voxel(p1.x(), p1.y(), p1.z(), v.getVoxelScale(), voxels.size() * 8 + borders.size() * 8));
		borders.push_back(Voxel(p2.x(), p2.y(), p2.z(), v.getVoxelScale(), voxels.size() * 8 + borders.size() * 8));
		borders.push_back(Voxel(p3.x(), p3.y(), p3.z(), v.getVoxelScale(), voxels.size() * 8 + borders.size() * 8));
		borders.push_back(Voxel(p4.x(), p4.y(), p4.z(), v.getVoxelScale(), voxels.size() * 8 + borders.size() * 8));
	}
	for (int y = 0; y < volumeSize.x(); ++y) {
		Eigen::Vector3d p1 = (v.getOrigin() + Eigen::Vector3d(0, y, 0) * v.getVoxelScale());
		Eigen::Vector3d p2 = (v.getOrigin() + Eigen::Vector3d(0, y, volumeSize.z()) * v.getVoxelScale());
		Eigen::Vector3d p3 = (v.getOrigin() + Eigen::Vector3d(volumeSize.x(), y, volumeSize.z()) * v.getVoxelScale());
		Eigen::Vector3d p4 = (v.getOrigin() + Eigen::Vector3d(volumeSize.x(), y, 0) * v.getVoxelScale());
		borders.push_back(Voxel(p1.x(), p1.y(), p1.z(), v.getVoxelScale(), voxels.size() * 8 + borders.size() * 8));
		borders.push_back(Voxel(p2.x(), p2.y(), p2.z(), v.getVoxelScale(), voxels.size() * 8 + borders.size() * 8));
		borders.push_back(Voxel(p3.x(), p3.y(), p3.z(), v.getVoxelScale(), voxels.size() * 8 + borders.size() * 8));
		borders.push_back(Voxel(p4.x(), p4.y(), p4.z(), v.getVoxelScale(), voxels.size() * 8 + borders.size() * 8));
	}
	for (int z = 0; z < volumeSize.x(); ++z) {
		Eigen::Vector3d p1 = (v.getOrigin() + Eigen::Vector3d(0, 0, z) * v.getVoxelScale());
		Eigen::Vector3d p2 = (v.getOrigin() + Eigen::Vector3d(0, volumeSize.y(), z) * v.getVoxelScale());
		Eigen::Vector3d p3 = (v.getOrigin() + Eigen::Vector3d(volumeSize.x(), volumeSize.y(), z) * v.getVoxelScale());
		Eigen::Vector3d p4 = (v.getOrigin() + Eigen::Vector3d(volumeSize.x(), 0, z) * v.getVoxelScale());
		borders.push_back(Voxel(p1.x(), p1.y(), p1.z(), v.getVoxelScale(), voxels.size() * 8 + borders.size() * 8));
		borders.push_back(Voxel(p2.x(), p2.y(), p2.z(), v.getVoxelScale(), voxels.size() * 8 + borders.size() * 8));
		borders.push_back(Voxel(p3.x(), p3.y(), p3.z(), v.getVoxelScale(), voxels.size() * 8 + borders.size() * 8));
		borders.push_back(Voxel(p4.x(), p4.y(), p4.z(), v.getVoxelScale(), voxels.size() * 8 + borders.size() * 8));
	}

	// Write header.
	outFile << "COFF" << std::endl;
	outFile << voxels.size() * 8 + borders.size() * 8 << " " << voxels.size() * 6 + borders.size() * 6 << " 0" << std::endl;

	for (size_t i = 0; i < voxels.size(); i++) {
		outFile << voxels[i].printVertices(colors[i]);
	}

	for (size_t i = 0; i < borders.size(); i++) {
		outFile << borders[i].printVertices(borderColor);
	}
	for (Voxel vo : voxels) {
		outFile << vo.printPlanes("255 0 0");
	}
	for (Voxel vo : borders) {
		outFile << vo.printPlanes("255 0 0");
	}

	outFile.close();
	return true;
}

bool Mesh::writeNormalMap(const std::shared_ptr<SurfaceMeasurement>& frame, const std::string& filename) {

	auto global_points = frame->getGlobalVertexMap();
	auto normal_map = frame->getGlobalNormalMap();

	std::cout << "Writing " << filename << std::endl;

	// Write off file.
	std::string filenameBaseOut = std::string("../result/mesh_");
	std::ofstream outFile(filenameBaseOut + filename + ".off");
	if (!outFile.is_open()) return false;

	outFile << "COFF" << std::endl;
	outFile << normal_map.size() << " " << "0" << " 0" << std::endl;

	for (size_t i = 0; i < normal_map.size(); i++) {
		const auto& vertex = global_points[i];
		const Vector3d& normal = 255 * (0.5 * normal_map[i] + Vector3d(0.5, 0.5, 0.5));

		if (vertex.allFinite() && normal.allFinite())
		{
			outFile << vertex.x() << " " << vertex.y() << " " << vertex.z() << " "
				<< int(normal.x()) << " " << int(normal.y()) << " " << int(normal.z()) << " " << " " << std::endl;
		}
		else {
			outFile << "0.0 0.0 0.0 0 0 0 0" << std::endl;
		}
	}

	// Close file.
	outFile.close();

	return true;
}

bool Mesh::writeMeshToFile(std::string filename, const std::shared_ptr<SurfaceMeasurement>& frame, bool includeCameraPose)
{
	auto global_points = frame->getGlobalVertexMap();
	auto color_map = frame->getColorMap();

	//Create Scene Mesh
	Mesh sceneMesh;
	for (size_t i = 0; i < global_points.size(); i++) {
		const auto& vertex = global_points[i];
		if (vertex.allFinite())
		{
			Vertex v;
			v.position = Vector4d{ vertex.x(), vertex.y(), vertex.z(), 1.f };
			v.color = Vector4uc{ color_map[i][0], color_map[i][1], color_map[i][2], color_map[i][3] };
			sceneMesh.addVertex(v);
		}
	}

	if (!includeCameraPose)
	{
		writeSurfaceMesh(sceneMesh, filename);
	}
	else
	{
		// Create Camera Mesh and add it to the Scene Mesh
		Mesh currentCameraMesh = Mesh::camera(frame->getGlobalPose(), 0.001f);
		writeSurfaceMesh(currentCameraMesh, "cam_" + filename);
		writeSurfaceMesh(sceneMesh, filename);
	}

	return true;
}

/**
 * Joins two meshes together by putting them into the common mesh and transforming the vertex positions of
 * mesh1 with transformation 'pose1to2'.
 */
Mesh Mesh::joinMeshes(const Mesh& mesh1, const Mesh& mesh2, Matrix4d pose1to2) {
	Mesh joinedMesh;
	const auto& vertices1 = mesh1.getVertices();
	const auto& triangles1 = mesh1.getTriangles();
	const auto& vertices2 = mesh2.getVertices();
	const auto& triangles2 = mesh2.getTriangles();

	auto& joinedVertices = joinedMesh.getVertices();
	auto& joinedTriangles = joinedMesh.getTriangles();

	const unsigned nVertices1 = vertices1.size();
	const unsigned nVertices2 = vertices2.size();
	joinedVertices.reserve(nVertices1 + nVertices2);

	const unsigned nTriangles1 = triangles1.size();
	const unsigned nTriangles2 = triangles2.size();
	joinedTriangles.reserve(nVertices1 + nVertices2);

	// Add all vertices (we need to transform vertices of mesh 1).
	for (int i = 0; i < nVertices1; ++i) {
		const auto& v1 = vertices1[i];
		Vertex v;
		v.position = pose1to2 * v1.position;
		v.color = v1.color;
		joinedVertices.push_back(v);
	}
	for (int i = 0; i < nVertices2; ++i) joinedVertices.push_back(vertices2[i]);

	// Add all faces (the indices of the second mesh need to be added an offset).
	for (int i = 0; i < nTriangles1; ++i) joinedTriangles.push_back(triangles1[i]);
	for (int i = 0; i < nTriangles2; ++i) {
		const auto& t2 = triangles2[i];
		Triangle t{ t2.idx0 + nVertices1, t2.idx1 + nVertices1, t2.idx2 + nVertices1 };
		joinedTriangles.push_back(t);
	}

	return joinedMesh;
}

/**
 * Generates a camera object with a given pose.
 */
Mesh Mesh::camera(const Matrix4d& cameraPose, float scale, Vector4uc color) {
	Mesh mesh;
	Matrix4d cameraToWorld = cameraPose.inverse();

	// These are precomputed values for sphere aproximation.
	std::vector<double> vertexComponents = { 25, 25, 0, -50, 50, 100, 49.99986, 49.9922, 99.99993, -24.99998, 25.00426, 0.005185,
		25.00261, -25.00023, 0.004757, 49.99226, -49.99986, 99.99997, -50, -50, 100, -25.00449, -25.00492, 0.019877 };
	const std::vector<unsigned> faceIndices = { 1, 2, 3, 2, 0, 3, 2, 5, 4, 4, 0, 2, 5, 6, 7, 7, 4, 5, 6, 1, 7, 1, 3, 7, 3, 0, 4, 7, 3, 4, 5, 2, 1, 5, 1, 6 };

	// Add vertices.
	for (int i = 0; i < 8; ++i) {
		Vertex v;
		v.position = cameraToWorld * Vector4d{ scale * float(vertexComponents[3 * i + 0]), scale * float(vertexComponents[3 * i + 1]), scale * float(vertexComponents[3 * i + 2]), 1.f };
		v.color = color;
		mesh.addVertex(v);
	}

	// Add faces.
	for (int i = 0; i < 12; ++i) {
		mesh.addFace(faceIndices[3 * i + 0], faceIndices[3 * i + 1], faceIndices[3 * i + 2]);
	}

	return mesh;
}
