// ============================================================================
//  
//  Project 1:   Lidar Obstacle Detection (Udacity Sensor Fusion Nanodegree)
// 
//  Authors:     Andreas Albrecht using code base/skeleton provided by Udacity
// 
//  Source:      https://github.com/udacity/SFND_Lidar_Obstacle_Detection.git
// 
//  			 Original source authored by Aaron Brown (Udacity)
//
// ============================================================================

// Helper functions for 3D Euclidean Clusering using KdTree search

#include "cluster3D.h"


// Helper function for recursive Euclidean clustering using a 3D KdTree search
void clusterHelper3D(int index, const std::vector<std::vector<float>>& points, std::vector<int>& cluster, std::vector<bool>& processed, KdTree3D* tree, float distanceTol)
{

	// Mark point given by index as processed
	processed[index] = true;
	cluster.push_back(index);

	// Search the 3D KdTree for the nearby neighbors within the distance tolerance
	std::vector<int> nearby = tree->search(points[index], distanceTol);

	// Iterate over all nearby neighbors
	for (int id : nearby)
	{
		// Pass current point to clusterHelper3D if it has not yet been processed
		if (!processed[id])
			clusterHelper3D(id, points, cluster, processed, tree, distanceTol);
	}

}


// Recursive Euclidean clustering using a 3D KdTree search
std::vector<std::vector<int>> euclideanCluster3D(const std::vector<std::vector<float>>& points, KdTree3D* tree, float distanceTol)
{

	// This function returns a list of indices for each target objct cluster found using 3D KdTree search

    // Create vector for all cluster index vectors (list of lists) 
	std::vector<std::vector<int>> clusters;

	// Create vector of booleans to track which points have already been processed (init all false => not processed yet)
	std::vector<bool> processed(points.size(), false);

	int id = 0;
	while (id < points.size())
	{
		// Check if current point has already been processed
		if (processed[id])
		{
			// Increment counter and skip the rest of the while loop statements
			id++;
			continue;
		}

        // Create vector for the current cluster index
		std::vector<int> cluster;

		// Recursive call of cluster helper function to find cluster by cluster
		clusterHelper3D(id, points, cluster, processed, tree, distanceTol);
		clusters.push_back(cluster);
		id++;
	}
 
	// Return vector of cluster indices
	return clusters;

}