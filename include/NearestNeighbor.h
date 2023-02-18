#pragma once

//#define FLANN_USE_CUDA
#include <flann/flann.hpp>
#include "Eigen.h"

using namespace std;

struct Match {
	int idx;
	float weight;
};

class NearestNeighborSearch {
public:
	NearestNeighborSearch() :
		m_maxDistance{ 0.005f },
		m_index{ nullptr },
		m_flatPoints{ nullptr }
	{ }

	~NearestNeighborSearch() {
		if (m_index) {
			delete m_flatPoints;
			delete m_index;
			m_flatPoints = nullptr;
			m_index = nullptr;
		}
	}

	void SetMatchingDistance(float maxDistance) {
		m_maxDistance = maxDistance;
	}

	void buildIndex(const vector<Eigen::Vector3d>& targetPoints) {
		m_flatPoints = new float[targetPoints.size() * 3];
		for (size_t pointIndex = 0; pointIndex < targetPoints.size(); pointIndex++) {
			for (size_t dim = 0; dim < 3; dim++) {
				m_flatPoints[pointIndex * 3 + dim] = targetPoints[pointIndex][dim];
			}
		}

		flann::Matrix<float> dataset(m_flatPoints, targetPoints.size(), 3);

		m_index = new flann::Index<flann::L2<float>>(dataset, flann::KDTreeIndexParams(1));
		//m_index = new flann::Index<flann::L2<float>>(dataset, flann::KDTreeCuda3dIndexParams());
		m_index->buildIndex();

		cout << "FLANN index created." << endl;
	}

	vector<Match> queryMatches(const vector<Vector3d>& transformedPoints) {
		if (!m_index) {
			cout << "FLANN index needs to be build before querying any matches." << endl;
			return {};
		}

		float* queryPoints = new float[transformedPoints.size() * 3];
		for (size_t pointIndex = 0; pointIndex < transformedPoints.size(); pointIndex++) {
			for (size_t dim = 0; dim < 3; dim++) {
				queryPoints[pointIndex * 3 + dim] = transformedPoints[pointIndex][dim];
			}
		}

		flann::Matrix<float> query(queryPoints, transformedPoints.size(), 3);
		flann::Matrix<int> indices(new int[query.rows * 1], query.rows, 1);
		flann::Matrix<float> distances(new float[query.rows * 1], query.rows, 1);

		flann::SearchParams searchParams{ 16 };
		searchParams.cores = 0;
		m_index->knnSearch(query, indices, distances, 1, searchParams);

		const unsigned nMatches = transformedPoints.size();
		vector<Match> matches;
		matches.reserve(nMatches);

		for (int i = 0; i < nMatches; ++i) {
			if (*distances[i] <= m_maxDistance)
				matches.push_back(Match{ *indices[i], 1.f });
			else
				matches.push_back(Match{ -1, 0.f });
		}

		delete[] query.ptr();
		delete[] indices.ptr();
		delete[] distances.ptr();

		return matches;
	}

private:
	float m_maxDistance;
	flann::Index<flann::L2<float>>* m_index;
	float* m_flatPoints;
};
