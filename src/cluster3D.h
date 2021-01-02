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

#ifndef CLUSTER3D_H
#define CLUSTER3D_H

#include "render/render.h"
#include "render/box.h"
#include <chrono>
#include <string>
#include "kdtree3D.h"

void clusterHelper3D(int index, const std::vector<std::vector<float>>& points, std::vector<int>& cluster, std::vector<bool>& processed, KdTree3D* tree, float distanceTol);

std::vector<std::vector<int>> euclideanCluster3D(const std::vector<std::vector<float>>& points, KdTree3D* tree, float distanceTol);

#endif