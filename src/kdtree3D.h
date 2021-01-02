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

// 3D KdTree implementation

#ifndef KDTREE3D_H
#define KDTREE3D_H

#include "render/render.h"


// Structure to represent a node of a 3D KdTree
struct Node
{
	// Node attributes: Point vector, node id, pointer to left and right neighbor nodes
	std::vector<float> point;
	int id;
	Node* left;
	Node* right;

	// Create new node passing in point array and node id and set left and right neighbor nodes to NULL by default
	Node(std::vector<float> arr, int setId)
	:	point(arr), id(setId), left(NULL), right(NULL)
	{}
};


// Structure of the 3D KdTree
struct KdTree3D
{
	// Define pointer to root note of 3D KdTree
	Node* root;

	// Instantiate 3D KdTree with root pointing NULL
	KdTree3D()
	: root(NULL)
	{}

	// Helper function to insert nodes to the 3D KdTree recursively
	// This function expects a double pointer to the next node, the current depth of the tree and the next point with id
	void insertHelper(Node** node, uint depth, std::vector<float> point, int id)
	{
		// Check if KdTree is empty or if the end of a branch has been reached (both linked to NULL pointer)
		if ((*node) == NULL)
		{
			// Insert a new node at the end of a branch (both left and right neigbhor nodes are initilized with NULL pointers)
			(*node) = new Node(point, id);
			/* Output for debugging
			std::cout << "Insert new node:" << std::endl;
			std::cout << "(*node)->id = " << (*node)->id << std::endl;
			std::cout << "(*node)->point = {" << (*node)->point[0] << ", " << (*node)->point[1] << ", " << (*node)->point[2] << "}" << std::endl;
			*/
		}
		else
		{
			// Traverse the 3D KdTree

			// Calculate current dimension of the tree to decide whether to split x, y or z dimension
			// depth % 3 = 0 => x-split
			// depth % 3 = 1 => y-split
			// depth % 3 = 2 => z-split
			uint currentDim = depth % 3;

			// Call insertHelper recursively in order to traverse the 3D KdTree terminating when hitting a NULL node
			if (point[currentDim] < ((*node)->point[currentDim]))
				// Follow the left branch to insert the next node and increment depth by 1
				insertHelper(&((*node)->left), depth+1, point, id);
			else
				// Follow the right branch to insert the next node and increment depth by 1
				insertHelper(&((*node)->right), depth+1, point, id);
		}
	}

	void insert(std::vector<float> point, int id)
	{
		// This function inserts a new point into the 3D KdTree starting from root

		// Initial depth to start KdTree insertion
		uint depth = 0;

		// Call recursive helper function => Start by passing in the memory address of root node, 
		// the depth of the tree (begin with depth = 0), the point value and the point id
		insertHelper(&root, depth, point, id);

	}

	// Helper function to perform a 3D KdTree search recursively
	void searchHelper(std::vector<float> target, Node* node, uint depth, float distanceTol, std::vector<int>& ids)
	{

		// Continue recursive search if no end of a branch (stop criterion) has yet been reached
		if (node != NULL)
		{
			// Check if the current point is located within a distance tolerance cube box around the target point
			if (
				(node->point[0] >= (target[0] - distanceTol) && node->point[0] <= (target[0] + distanceTol)) 
				&& (node->point[1] >= (target[1] - distanceTol) && node->point[1] <= (target[1] + distanceTol))
				&& (node->point[2] >= (target[2] - distanceTol) && node->point[2] <= (target[2] + distanceTol))
				)
			{
				// Calculate the exact distance between the current point and the target point
				float distance = sqrt(
					(node->point[0] - target[0]) * (node->point[0] - target[0])
					+ (node->point[1] - target[1]) * (node->point[1] - target[1])
					+ (node->point[2] - target[2]) * (node->point[2] - target[2])
					);
				// Push current point to the list if it is located within a distance tolerance circle around the target
				if (distance <= distanceTol)
				{
					ids.push_back(node->id);
					/* Output for debugging
					std::cout << "Current node id pushed node ids list:" << std::endl;
					std::cout << "node->id = " << node->id << std::endl;
					std::cout << "node->point = {" << node->point[0] << ", " << node->point[1] << ", " << node->point[2] << "}" << std::endl;
					*/
				}
			}

			// Get current dimension of the tree to decide whether to search in x, y or z dimension
			// depth % 3 = 0 => search in x-direction
			// depth % 3 = 1 => search in y-direction
			// depth % 3 = 2 => search in z-direction
			uint currentDim = depth % 3;

			// Check across distance tolerance boundaries whether to branch the next search step to the left or right
			// (alternating between x-, y- and z-split according to the tree depth of the current search position)
			if ((target[currentDim] - distanceTol) < node->point[currentDim])
				// Follow the left branch to search the next node and increment depth by 1
				searchHelper(target, node->left, depth+1, distanceTol, ids);
			if ((target[currentDim] + distanceTol) > node->point[currentDim])
				// Follow the right branch to search the next node and increment depth by 1
				searchHelper(target, node->right, depth+1, distanceTol, ids);
		}

	}

	// Return a list of point ids in the 3D KdTree that are within the distance tolerance of the target point
	std::vector<int> search(std::vector<float> target, float distanceTol)
	{
		// Initialize list of point ids
		std::vector<int> ids;

		// Set initial depth of the  search to zero
		uint depth = 0;

		// Call recursive searchHelper function 
		searchHelper(target, root, depth, distanceTol, ids);

		// Return list of points located within distance tolerance of the target point
		return ids;
	}
	
};

#endif