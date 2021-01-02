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

// PCL lib Functions for 3D Lidar point cloud processing

#include "processPointClouds.h"


//constructor:
template<typename PointT>
ProcessPointClouds<PointT>::ProcessPointClouds() {}


//de-constructor:
template<typename PointT>
ProcessPointClouds<PointT>::~ProcessPointClouds() {}


// Count number of points in point cloud:
template<typename PointT>
void ProcessPointClouds<PointT>::numPoints(typename pcl::PointCloud<PointT>::Ptr cloud)
{
    std::cout << cloud->points.size() << std::endl;
}


// Point cloud filter for downsampling using voxel grid approach and 3D region of interest cropping
template<typename PointT>
typename pcl::PointCloud<PointT>::Ptr ProcessPointClouds<PointT>::FilterCloud(typename pcl::PointCloud<PointT>::Ptr cloud, float filterRes, Eigen::Vector4f minPoint, Eigen::Vector4f maxPoint)
{

    // This function performs a voxel grid point reduction and region based filtering

    // Time segmentation process
    auto startTime = std::chrono::steady_clock::now();

    // Create a cloud object to hold the downsampled voxel grid point cloud
    typename pcl::PointCloud<PointT>::Ptr cloudFiltered (new pcl::PointCloud<PointT>);

    // Create the cube-spaced voxel grid filtering object: Downsample the dataset using a leaf size 
    // of filterRes x filterRes x filterRes in x-, y-, and z-direction (e. g. filterRes = 0.2 m)
    pcl::VoxelGrid<PointT> vg;
    //std::cout << typeid(vg).name() << endl;
    vg.setInputCloud(cloud);
    vg.setLeafSize(filterRes, filterRes, filterRes);
    vg.filter(*cloudFiltered);

    // Create a cloud object to hold the region of interest resp. the ego vehicle's field of view to
    // be cropped from the downsampled point cloud for further processing (e. g. object detection)
    typename pcl::PointCloud<PointT>::Ptr cloudRegion (new pcl::PointCloud<PointT>);

    // Create a 3D crop box filtering object to pick the region of interest of the point cloud
    pcl::CropBox<PointT> region(true);
    region.setMin(minPoint);
    region.setMax(maxPoint);
    region.setInputCloud(cloudFiltered);
    region.filter(*cloudRegion);

    // Initialize vecotr of indices of Lidar points that belong to the ego vehicle's roof
    std::vector<int> indices;

    // Define coordinates of crop box corner points to remove the Lidar locations on the ego vehicle's roof
    float xmin_crop = -1.5;
    float ymin_crop = -1.7;
    float zmin_crop = -1.0;
    float xmax_crop = 2.6;
    float ymax_crop = 1.7;
    float zmax_crop = -0.4;
    
    // Filter to remove the Lidar locations on the ego vehicle's roof from the region of interest cloud
    pcl::CropBox<PointT> roof(true);
    // Initialize quaternion representation of crop box corner points (q = 1)
    roof.setMin(Eigen::Vector4f (xmin_crop, ymin_crop, zmin_crop, 1));
    roof.setMax(Eigen::Vector4f (xmax_crop, ymax_crop, zmax_crop, 1));
    roof.setInputCloud(cloudRegion);
    roof.filter(indices);  // Get indices of roof points

    // Get all inliers that belong to the ego vehicle's roof
    pcl::PointIndices::Ptr inliers {new pcl::PointIndices};
    for (int point : indices)
        inliers->indices.push_back(point);

    // Extract / Remove indices of the Lidar points on the ego vehicle's roof from the cropped cloud region
    pcl::ExtractIndices<PointT> extract;
    extract.setInputCloud(cloudRegion);
    extract.setIndices(inliers);  // Set indices of the inliers that belong to the ego vehicle's roof
    extract.setNegative(true);  // Set remove indices of the inliers
    extract.filter(*cloudRegion);  // apply filter on cropped cloud region
    
    auto endTime = std::chrono::steady_clock::now();
    //auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    //std::cout << "filtering took " << elapsedTime.count() << " milliseconds" << std::endl;
    auto elapsedTime = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);
    std::cout << "filtering took " << elapsedTime.count() << " microseconds" << std::endl;

    // Return the cropped cloud region
    return cloudRegion;

}


// Function to separate point cloud clusters of the obstcles from the point cluster of the driving ground (plane / road)
template<typename PointT>
std::pair<typename pcl::PointCloud<PointT>::Ptr, typename pcl::PointCloud<PointT>::Ptr> ProcessPointClouds<PointT>::SeparateClouds(pcl::PointIndices::Ptr inliers, typename pcl::PointCloud<PointT>::Ptr cloud) 
{
    // Create two new point clouds, one cloud with obstacles and other with segmented plane
    typename pcl::PointCloud<PointT>::Ptr obstacleCloud (new pcl::PointCloud<PointT>());
    typename pcl::PointCloud<PointT>::Ptr planeCloud (new pcl::PointCloud<PointT>());

    // Loop over all inlier points of the original point cloud and push them to the plane cloud object
    for (int index : inliers->indices) {
        planeCloud->points.push_back(cloud->points[index]);
    }
    
    // Get obstacle cloud
    pcl::ExtractIndices<PointT> extract;
    extract.setInputCloud(cloud);
    extract.setIndices(inliers);
    extract.setNegative(true);
    extract.filter(*obstacleCloud);

    // Pair obstacle cloud and plane cloud
    std::pair<typename pcl::PointCloud<PointT>::Ptr, typename pcl::PointCloud<PointT>::Ptr> segResult(obstacleCloud, planeCloud);

    return segResult;
}


// Function to segment the point cluster that belongs to the driving ground (plane / road)
template<typename PointT>
std::pair<typename pcl::PointCloud<PointT>::Ptr, typename pcl::PointCloud<PointT>::Ptr> ProcessPointClouds<PointT>::SegmentPlane(typename pcl::PointCloud<PointT>::Ptr cloud, int maxIterations, float distanceThreshold)
{
    // Time segmentation process
    auto startTime = std::chrono::steady_clock::now();

    // This function separates a given point cloud into two parts, the drivable plane and obstacles moving on the drivable plane
    // 
    // Perform plane-from-obstacle point segmentation using random sample consensus method from point cloud libarary:
    // See e.g.
    // - https://pcl-tutorials.readthedocs.io/en/master/planar_segmentation.html
    // - https://pointcloudlibrary.github.io/documentation/classpcl_1_1_s_a_c_segmentation.html

    // Prepare segmentation: Create object to hold the inliers
    pcl::PointIndices::Ptr inliers {new pcl::PointIndices()};

    // Prepare segmentation: Create object to hold the model coefficients
    pcl::ModelCoefficients::Ptr coefficients {new pcl::ModelCoefficients()};

    // Create the segmentation object using PointT template as input argument to pass in different input arguments lateron
    pcl::SACSegmentation<PointT> seg;

    // Optional: Optimize geometric model coefficients
    seg.setOptimizeCoefficients(true);

    // Mandatory: Select optimization parameters to fit the points to the target model (here: plane) geometry
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);  // RANSAC = Random Sample Consensus Algorithm
    seg.setMaxIterations(maxIterations);
    seg.setDistanceThreshold(distanceThreshold);

    // Segment the largest planar component from the input point cloud
    seg.setInputCloud(cloud);
    seg.segment(*inliers, *coefficients);
    if (inliers->indices.size () == 0)
    {
        std::cerr << "Could not estimate a planar model for the given dataset." << std::endl;
    }

    // Get two separated clouds from identified inliers and original point cloud
    std::pair<typename pcl::PointCloud<PointT>::Ptr, typename pcl::PointCloud<PointT>::Ptr> segResult = SeparateClouds(inliers, cloud);
    
    // Stop timer
    auto endTime = std::chrono::steady_clock::now();
    //auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    //std::cout << "plane segmentation took " << elapsedTime.count() << " milliseconds" << std::endl;
    auto elapsedTime = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);
    std::cout << "RansacPlane3D took " << elapsedTime.count() << " microseconds" << std::endl;

    // Return  of inliers from fitted line with most inliers
    return segResult;
}


/* SegmentPlaneRansac3D_v1 (Variant 1):
 * 
 * Function to segment a point cluster representing the drivable ground (plane) from point clusters representing obstacles
 * using a simple self-made Ransac optimization algorithm to fit a 3D plane model with a tolerance band:
 * 
 * 1) Randomly pick 3 points from the cloud and calculate a 3D plane model through these 3D key points.
 * 2) Loop through all 3D points of the overall point cloud and calculate the distance to the 3D plane for each of them
 * 3) If the distance is below a distance tolerance threshold, consider this point an inlier of drivable ground plane 
 *    and add it to a temporary set, else this point is considered an outlier, which is part of the obstacles above ground.
 * 4) Store the total (temporary) set of inliers if it is larger than the total set of inliers from the previous iteration.
 * 5) Repeat above steps until the maximum number of iterations is reached.
 */
template<typename PointT>
std::pair<typename pcl::PointCloud<PointT>::Ptr, typename pcl::PointCloud<PointT>::Ptr> ProcessPointClouds<PointT>::SegmentPlaneRansac3D_v1(typename pcl::PointCloud<PointT>::Ptr cloud, int maxIterations, float distanceThreshold)
{

    // Time segmentation process
    auto startTime = std::chrono::steady_clock::now();
	
	// Create unordered set holding the best (largest) set of inliers of the drivable ground found so far
	std::unordered_set<int> inlierIndicesResult;

    // Initialize random number generator
    srand(time(NULL));

    // Declare point and plane variables
    PointT next_point;
    float x1, x2, x3, x4, y1, y2, y3, y4, z1, z2, z3, z4;
    float v1x, v1y, v1z, v2x, v2y, v2z, v3x, v3y, v3z;
    float nx, ny, nz;
    float a, b, c, d;
    float dist;

	// For max iterations	
	while (maxIterations--)
    {
		// Create an empty temporary (unordered) set (containing each element only once) of inliers per iterations 
		std::unordered_set<int> inlierIndices;

		// Randomly sample subset and fit plane => Here: Randomly pick three points
		while (inlierIndices.size() < 3)
        {
			// Randomly pick a point from point cloud and insert it to the temporary set of inliers
			// Find the next point in memory as random number >= 0 modulo point size
			inlierIndices.insert((rand() % cloud->points.size()));
		}

		// Get a pointer to the first plane key point (x1, y1, z1) in the temporary set of inliers
		auto itr = inlierIndices.begin();
		x1 = cloud->points[*itr].x;
		y1 = cloud->points[*itr].y;
		z1 = cloud->points[*itr].z;
		// Increment pointer and get the second plane key point (x2, y2, z2) in the temporary set of inliers
		itr++;
		x2 = cloud->points[*itr].x;
		y2 = cloud->points[*itr].y;
		z2 = cloud->points[*itr].z;
		// Increment pointer and get the third plane key point (x3, y3, z3) in the temporary set inliers
		itr++;
		x3 = cloud->points[*itr].x;
		y3 = cloud->points[*itr].y;
		z3 = cloud->points[*itr].z;

		// Vector v1 = [x2 - x1, y2 - y1, z2 - z1]
		v1x = x2 - x1;
		v1y = y2 - y1;
		v1z = z2 - z1;

		// Vector v2 = [x3 - x1, y3 - y1, z3 - z1]
		v2x = x3 - x1;
		v2y = y3 - y1;
		v2z = z3 - z1;

		// Normal vector of the plane n = v1 x v2
		nx = v1y * v2z - v1z * v2y;
		ny = v1z * v2x - v1x * v2z;
		nz = v1x * v2y - v1y * v2x;

		// Parametric plane equation in 3D space
		// a * x + b * y + c * z + d = 0
		a = nx;
		b = ny;
		c = nz;
		d = -(nx * x1 + ny * y1 + nz * z1);

		// Loop over all points in the point cloud
		for (int cnt = 0; cnt < cloud->points.size(); cnt++)
        {
			// Check if the current point (current index) is already a member of our temporary set of inliers
			if (inlierIndices.count(cnt) > 0)
            {
				// If it is than skip next section and continue with next increment of the for-loop
				continue;
			}

			// Add the current point (x4, y4, z4) if it is not yet part of our (unique) temporary set of inliers of the plane
			next_point = cloud->points[cnt];
			x4 = next_point.x;
			y4 = next_point.y;
			z4 = next_point.z;

			// Measure the distance for each point in the point cloud to the 3D plane (in normal direction)
			dist = fabs(a * x4 + b * y4 + c * z4 + d) / sqrt(a * a + b * b + c * c);

			// If the distance is smaller than the distance tolerance threshold add the point to the temporary set of inliers of the plane
			if (dist <= distanceThreshold)
            {
				inlierIndices.insert(cnt);
			}
		}

		// Check if the current set of inliers is larger than the set of inliers from the previous iteration
		if (inlierIndices.size() > inlierIndicesResult.size())
        {
            // Use the the current set of inliers if it is larger than the previous one
            inlierIndicesResult.clear();
			inlierIndicesResult = inlierIndices;
		}
	}

    // Check if inliers have been found
    if (inlierIndicesResult.size () == 0)
    {
        std::cerr << "Could not estimate a planar model for the given dataset." << std::endl;
    }

    // Initialize an inlier point cloud for the plane and an outlier point cloud for the the obstacles
    typename pcl::PointCloud<PointT>::Ptr outlierCloud (new pcl::PointCloud<PointT>());
    typename pcl::PointCloud<PointT>::Ptr inlierCloud (new pcl::PointCloud<PointT>());

    // Split the point cloud into an inlier cloud for the drivable ground (plane) and an outlier cloud for the obstacles above ground
    for(int index = 0; index < cloud->points.size(); index++)
    {
        next_point = cloud->points[index];
        if(inlierIndicesResult.count(index))
            inlierCloud->points.push_back(next_point);
        else
            outlierCloud->points.push_back(next_point);
    }

    // Pair oulier cloud for the obstacles and the inlier cloud for the drivable ground (plane)
    std::pair<typename pcl::PointCloud<PointT>::Ptr, typename pcl::PointCloud<PointT>::Ptr> segResult(outlierCloud, inlierCloud);

    // Stop timer
    auto endTime = std::chrono::steady_clock::now();
    //auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    //std::cout << "plane segmentation took " << elapsedTime.count() << " milliseconds" << std::endl;
    auto elapsedTime = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);
    std::cout << "RansacPlane3D took " << elapsedTime.count() << " microseconds" << std::endl;

	// Return  of inliers from fitted line with most inliers
	return segResult;

}


/* SegmentPlaneRansac3D_v2 (Variant 2):
 * 
 * Function to segment a point cluster representing the drivable ground (plane) from point clusters representing obstacles
 * using a simple self-made Ransac optimization algorithm to fit a 3D plane model with a tolerance band:
 * 
 * 1) Randomly pick 3 points from the cloud and calculate a 3D plane model through these 3D key points.
 * 2) Loop through all 3D points of the overall point cloud and calculate the distance to the 3D plane for each of them
 * 3) If the distance is below a distance tolerance threshold, consider this point an inlier of drivable ground plane 
 *    and add it to a temporary set, else this point is considered an outlier, which is part of the obstacles above ground.
 * 4) Store the total (temporary) set of inliers if it is larger than the total set of inliers from the previous iteration.
 * 5) Repeat above steps until the maximum number of iterations is reached.
 */
template<typename PointT>
std::pair<typename pcl::PointCloud<PointT>::Ptr, typename pcl::PointCloud<PointT>::Ptr> ProcessPointClouds<PointT>::SegmentPlaneRansac3D_v2(typename pcl::PointCloud<PointT>::Ptr cloud, int maxIterations, float distanceThreshold)
{
    // Time segmentation process
    auto startTime = std::chrono::steady_clock::now();

    // Create unordered set holding the best (largest) set of inliers of the drivable ground found so far
    /*Buffer to hold the indices of the points within distanceTol , it shall hold max identified indices*/
	std::unordered_set<int> inlierIndicesResult;

    // Initialize random number generator
	srand(time(NULL));

    // Declare point and plane variables
	PointT point_p1, point_p2, point_p3, point_p4, next_point;
    float v1x, v1y, v1z, v2x, v2y, v2z, v3x, v3y, v3z;
    float nx, ny, nz;
    float a, b, c, d;
    float dist;

    // For max iterations	
	while (maxIterations--)
    {
		// Create an empty temporary (unordered) set (containing each element only once) of inliers per iterations 
		std::unordered_set<int> inlierIndices;

		// Randomly sample subset and fit plane => Here: Randomly pick three points
		while (inlierIndices.size() < 3)
        {
			// Randomly pick a point from point cloud and insert it to the temporary set of inliers
			// Find the next point in memory as random number >= 0 modulo point size
			inlierIndices.insert((rand() % cloud->points.size()));
		}

        // Get a pointer to the first plane key point p1 in the temporary set of inliers
		auto itr = inlierIndices.begin();
        point_p1 = cloud->points[*itr];
		// Increment pointer and get the second plane key point p2 in the temporary set of inliers
		itr++;
		point_p2 = cloud->points[*itr];
		// Increment pointer and get the third plane key point p3 in the temporary set inliers
		itr++;
		point_p3 = cloud->points[*itr];

        // Vector v1 = p2 - p1
		v1x = point_p2.x - point_p1.x;
		v1y = point_p2.y - point_p1.y;
		v1z = point_p2.z - point_p1.z;

		// Vector v2 = p3 - p1
		v2x = point_p3.x - point_p1.x;
		v2y = point_p3.y - point_p1.y;
		v2z = point_p3.z - point_p1.z;

		// Normal vector of the plane n = v1 x v2
		nx = v1y * v2z - v1z * v2y;
		ny = v1z * v2x - v1x * v2z;
		nz = v1x * v2y - v1y * v2x;

		// Parametric plane equation in 3D space
		// a * x + b * y + c * z + d = 0
		a = nx;
		b = ny;
		c = nz;
		d = -(nx * point_p1.x + ny * point_p1.y + nz * point_p1.z);

		// Loop over all points in the point cloud
		for (int cnt = 0; cnt < cloud->points.size(); cnt++)
        {
			// Check if the current point (current index) is already a member of our temporary set of inliers
			if (inlierIndices.count(cnt) > 0)
            {
				// If it is than skip next section and continue with next increment of the for-loop
				continue;
			}

			// Add the current point p4 if it is not yet part of our (unique) temporary set of inliers of the plane
			point_p4 = cloud->points[cnt];

			// Measure the distance for each point in the point cloud to the 3D plane (in normal direction)
			dist = fabs(a * point_p4.x + b * point_p4.y + c * point_p4.z + d) / sqrt(a * a + b * b + c * c);

			// If the distance is smaller than the distance tolerance threshold add the point to the temporary set of inliers of the plane
			if (dist <= distanceThreshold)
            {
				inlierIndices.insert(cnt);
			}
		}

		// Check if the current set of inliers is larger than the set of inliers from the previous iteration
		if (inlierIndices.size() > inlierIndicesResult.size())
        {
            // Use the the current set of inliers if it is larger than the previous one
            inlierIndicesResult.clear();
			inlierIndicesResult = inlierIndices;
		}
	}

    // Check if inliers have been found
    if (inlierIndicesResult.size () == 0)
    {
        std::cerr << "Could not estimate a planar model for the given dataset." << std::endl;
    }

    // Initialize an inlier point cloud for the plane and an outlier point cloud for the the obstacles
    typename pcl::PointCloud<PointT>::Ptr outlierCloud (new pcl::PointCloud<PointT>());
    typename pcl::PointCloud<PointT>::Ptr inlierCloud (new pcl::PointCloud<PointT>());

    // Split the point cloud into an inlier cloud for the drivable ground (plane) and an outlier cloud for the obstacles above ground
    for(int index = 0; index < cloud->points.size(); index++)
    {
        next_point = cloud->points[index];
        if(inlierIndicesResult.count(index))
            inlierCloud->points.push_back(next_point);
        else
            outlierCloud->points.push_back(next_point);
    }

    // Pair oulier cloud for the obstacles and the inlier cloud for the drivable ground (plane)
    std::pair<typename pcl::PointCloud<PointT>::Ptr, typename pcl::PointCloud<PointT>::Ptr> segResult(outlierCloud, inlierCloud);

    // Stop timer
    auto endTime = std::chrono::steady_clock::now();
    //auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    //std::cout << "plane segmentation took " << elapsedTime.count() << " milliseconds" << std::endl;
    auto elapsedTime = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);
    std::cout << "RansacPlane3D took " << elapsedTime.count() << " microseconds" << std::endl;

	// Return  of inliers from fitted line with most inliers
	return segResult;

}


// Function to perform Euclidean clustering using Point Cloud Library methods
template<typename PointT>
std::vector<typename pcl::PointCloud<PointT>::Ptr> ProcessPointClouds<PointT>::Clustering(typename pcl::PointCloud<PointT>::Ptr cloud, float clusterTolerance, int minSize, int maxSize)
{

    // This function performs 3D euclidean clustering to group detected obstacles using KdTree search method

    // Time clustering process
    auto startTime = std::chrono::steady_clock::now();

    // Create a vector of point cloud clusters to store the detected object clusters
    std::vector<typename pcl::PointCloud<PointT>::Ptr> clusters;

    // Create a KdTree object to enhance efficient search of clusters
    typename pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>);

    // Feed in the cloud into the KdTree
    tree->setInputCloud(cloud);

    // Create cluster indices and euclidean cluster extraction object
    std::vector<pcl::PointIndices> clusterIndices;
    pcl::EuclideanClusterExtraction<PointT> ec;
    ec.setClusterTolerance(clusterTolerance);
    ec.setMinClusterSize(minSize);
    ec.setMaxClusterSize(maxSize);
    ec.setSearchMethod(tree);
    ec.setInputCloud(cloud);
    ec.extract(clusterIndices);

    // Loop over all cluster indices
    for (pcl::PointIndices getIndices: clusterIndices)
    {
        // Create a new cloud cluster
        typename pcl::PointCloud<PointT>::Ptr cloudCluster(new pcl::PointCloud<PointT>);

        // Iterate over the indices and push them back to the obstacle cloud cluster
        for (int index : getIndices.indices)
            cloudCluster->points.push_back(cloud->points[index]);
        
        cloudCluster->width = cloudCluster->points.size();
        cloudCluster->height = 1;
        cloudCluster->is_dense = true;

        clusters.push_back(cloudCluster);

    }

    auto endTime = std::chrono::steady_clock::now();
    //auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    //std::cout << "clustering took " << elapsedTime.count() << " milliseconds and found " << clusters.size() << " clusters" << std::endl;
    auto elapsedTime = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);
    std::cout << "clustering took " << elapsedTime.count() << " microseconds and found " << clusters.size() << " clusters" << std::endl;

    return clusters;
}


// Function to perform Euclidean Clustering using 3D KdTree to groud detected obstacles
template<typename PointT>
std::vector<typename pcl::PointCloud<PointT>::Ptr> ProcessPointClouds<PointT>::ClusteringKdTree3D(typename pcl::PointCloud<PointT>::Ptr cloud, float clusterTolerance, int minSize, int maxSize)
{

    // Time clustering process
    auto startTime = std::chrono::steady_clock::now();

    // Init vector of point cloud clusters to be returned as output argument
    std::vector<typename pcl::PointCloud<PointT>::Ptr> clusters;

    // Create a 3D KdTree object to enhance efficient search for clusters
	KdTree3D* tree = new KdTree3D;
  
	// Loop over all points in point cloud and insert them one by one as 3D vectors into the KdTree
    std::vector<std::vector<float>> pointVectors;
    //std::vector<float> pointVector;
    for (int i=0; i < cloud->points.size(); i++)
    {
        // Get x-, y- and z-component of current cloud point and push them to vector structure
        std::vector<float> pointVector({cloud->points[i].x, cloud->points[i].y, cloud->points[i].z});
        //pointVector.push_back(cloud->points[i].x);
        //pointVector.push_back(cloud->points[i].y);
        //pointVector.push_back(cloud->points[i].z);

        /* Output for debugging
        std::cout << "pointVector = {" << pointVector[0] << ", " << pointVector[1] << ", " << pointVector[2] << "}" << std::endl;
        */

        // Insert new point into the 3D KdTree
    	tree->insert(pointVector,i);

        // Push new point to vector of point vectors
        pointVectors.push_back(pointVector);
    }

    // Find clusters (by point indices) using Euclidean clustering and the given cluster tolerance
    std::vector<std::vector<int>> clustersIndices = euclideanCluster3D(pointVectors, tree, clusterTolerance);

    // Loop through the indices of all clusters
    for (std::vector<int> clusterIndex : clustersIndices)
  	{
        // Create new point cloud cluster
        typename pcl::PointCloud<PointT>::Ptr clusterCloud(new pcl::PointCloud<PointT>());

        // Loop through all point indices in the current cluster
  		for (int index : clusterIndex)
        {
            clusterCloud->points.push_back(cloud->points[index]);
        }
        clusterCloud->width = clusterCloud->points.size();
        clusterCloud->height = 1;
        clusterCloud->is_dense = true;
        // Keep current point cloud cluster if the number of member points is within the specified limits
        if ((minSize <= clusterCloud->width) and (clusterCloud->width <= maxSize))
        {
            // Push cluster into output argument
            clusters.push_back(clusterCloud);
        }
  	}

    // Stop timer and calculate elapsed clustering process time
    auto endTime = std::chrono::steady_clock::now();
    //auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    //std::cout << "clustering took " << elapsedTime.count() << " milliseconds and found " << clusters.size() << " clusters" << std::endl;
    auto elapsedTime = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);
    std::cout << "clustering took " << elapsedTime.count() << " microseconds and found " << clusters.size() << " clusters" << std::endl;

    // Return all point cloud clusters found where minSize <= clusters[id] <= maxSize for each id
    return clusters;

}


// Function to calculate the best fit 3D bounding box position around a given point cloud cluster
template<typename PointT>
Box ProcessPointClouds<PointT>::axisAlignedBoundingBox(typename pcl::PointCloud<PointT>::Ptr cluster)
{

    // Find bounding box for one of the clusters using the outermost points of the cluster in x-, y- and z-direction
    PointT minPoint, maxPoint;
    pcl::getMinMax3D(*cluster, minPoint, maxPoint);

    Box box;
    box.x_min = minPoint.x;
    box.y_min = minPoint.y;
    box.z_min = minPoint.z;
    box.x_max = maxPoint.x;
    box.y_max = maxPoint.y;
    box.z_max = maxPoint.z;

    return box;
}


// Function to calculate an z-axis oriented 3D bounding box around a given point cloud cluster using Principal Component Analysis (PCA)
template<typename PointT>
BoxQ ProcessPointClouds<PointT>::orientedBoundingBox(typename pcl::PointCloud<PointT>::Ptr cluster)
{
    /*
    Find an optimum 3D bounding box for one of the clusters using PCA to also find the optimum orientation
    Reference: http://codextechnicanum.blogspot.com/2015/04/find-minimum-oriented-bounding-box-of.html

    Process to fit a 3D bouding box to a 3D point cluster with respect to its centroid position and a rotation around the z-axis:
    1) compute the centroid (c0, c1, c2) and the normalized covariance
    2) compute the eigenvectors e0, e1, e2. The reference system will be (e0, e1, e0 X e1) --- note: e0 X e1 = +/- e2
    3) move the points in that RF --- note: the transformation given by the rotation matrix (e0, e1, e0 X e1) & (c0, c1, c2) must be inverted
    4) compute the max, the min and the center of the diagonal
    5) given a box centered at the origin with size (max_pt.x - min_pt.x, max_pt.y - min_pt.y, max_pt.z - min_pt.z) the transformation you have to apply is Rotation = (e0, e1, e0 X e1) & Translation = Rotation * center_diag + (c0, c1, c2)

    // First find the eigenvectors for the covariance matrix of the point cloud (i.e. principal component analysis, PCA).
    // Replace the cloudSegmented pointer with a pointer to the cloud cluster you want to find the oriented bounding box for.
    
    Quaternions are a way to do rotations https://www.youtube.com/watch?v=mHVwd8gYLnI
    */

    // Define a 3D bounding box (quaternion representation)
    BoxQ box;

    // Test if cluster is 3-dimensional and not collinear or coplanar
    assert(cluster->points.size() >= 3);

    // Project cluster to x-y-plane in order to find the orientation angle only about the z-axis
    typename pcl::PointCloud<PointT>::Ptr clusterProjectedToXYPlane(new pcl::PointCloud<PointT>);
    pcl::copyPointCloud(*cluster, *clusterProjectedToXYPlane);
    for (auto &point : clusterProjectedToXYPlane->points)
    {
        // Suppress z-dimension
        point.z = 0;
    }

    // Compute principal directions of the point cluster projected to XY-plane (suppress rotations around x- and y-axis)
    Eigen::Vector4f pcaCentroid;
    pcl::compute3DCentroid(*clusterProjectedToXYPlane, pcaCentroid);
    Eigen::Matrix3f covariance;
    computeCovarianceMatrixNormalized(*clusterProjectedToXYPlane, pcaCentroid, covariance);
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigen_solver(covariance, Eigen::ComputeEigenvectors);
    Eigen::Matrix3f eigenVectorsPCA = eigen_solver.eigenvectors();
    // The next statement is necessary for proper orientation in some cases. The numbers come out the same without it, 
    // but the signs are different and the box doesn't get correctly oriented in some cases.
    eigenVectorsPCA.col(2) = eigenVectorsPCA.col(0).cross(eigenVectorsPCA.col(1));

    // Note that getting the eigenvectors can also be obtained via the PCL PCA interface with something like:
    //pcl::PointCloud<pcl::PointT>::Ptr cloudPCAprojection (new pcl::PointCloud<pcl::PointT>);
    //pcl::PCA<pcl::PointT> pca;
    //pca.setInputCloud(cluster);
    //pca.project(*cluster, *cloudPCAprojection);
    //pca.setInputCloud(clusterProjectedToXYPlane);
    //pca.project(*clusterProjectedToXYPlane, *cloudPCAprojection);
    //std::cerr << std::endl << "EigenVectors: " << pca.getEigenVectors() << std::endl;
    //std::cerr << std::endl << "EigenValues: " << pca.getEigenValues() << std::endl;
    // In this case, pca.getEigenVectors() gives similar eigenVectors to eigenVectorsPCA.
    //const auto pcaCentroid = pca.getMean();
    //const auto eigenVectorsPCA = pca.getEigenVectors();

    // These eigenvectors are used to transform the point cloud to the origin point (0, 0, 0) such that
    // the eigenvectors correspond to the axes of the space. The minimum point, maximum point, and the 
    // middle of the diagonal between these two points are calculated for the transformed cloud (also 
    // referred to as the projected cloud when using PCL's PCA interface). 

    // Transform the original cloud to the origin where the principal components correspond to the axes.
    Eigen::Matrix4f projectionTransform(Eigen::Matrix4f::Identity());
    projectionTransform.block<3,3>(0,0) = eigenVectorsPCA.transpose();
    projectionTransform.block<3,1>(0,3) = -1.f * (projectionTransform.block<3,3>(0,0) * pcaCentroid.head<3>());

    // Project point cluster to normalized space
    typename pcl::PointCloud<PointT>::Ptr clusterPointsProjected(new pcl::PointCloud<PointT>);
    pcl::transformPointCloud(*cluster, *clusterPointsProjected, projectionTransform);

    // Get the minimum and maximum points of the transformed cloud.
    PointT minPoint, maxPoint;
    pcl::getMinMax3D(*clusterPointsProjected, minPoint, maxPoint);
    const Eigen::Vector3f meanDiagonal = 0.5f*(maxPoint.getVector3fMap() + minPoint.getVector3fMap());
    
    // Get the dimensions of an axis-aligned 3D bounding box using the minimum and maximum points
    float cube_length = maxPoint.x - minPoint.x;
    float cube_width = maxPoint.y - minPoint.y;
    float cube_height = maxPoint.z - minPoint.z;

    // Final transformation: Calculate the quaternion using the eigenvectors from PCA, which determines how
    // the final 3D box is oriented about the z-axis, and the transform to pu the box in correct 3D position.
    Eigen::Quaternionf bboxQuaternion(eigenVectorsPCA);
    Eigen::Vector3f bboxTransform = eigenVectorsPCA * meanDiagonal + pcaCentroid.head<3>();

    // Set parameters of our oriented 3D bounding box
    box.bboxQuaternion = bboxQuaternion;
    box.bboxTransform = bboxTransform;
    box.cube_length = cube_length;
    box.cube_width = cube_width;
    box.cube_height = cube_height;

    // Return the optimum oriented 3D bounding box fit to the point cluster
    return box;
}


// Save point cloud data to user-specified file
template<typename PointT>
void ProcessPointClouds<PointT>::savePcd(typename pcl::PointCloud<PointT>::Ptr cloud, std::string file)
{
    pcl::io::savePCDFileASCII (file, *cloud);
    std::cerr << "Saved " << cloud->points.size () << " data points to "+file << std::endl;
}


// Load point cloud data file from given path
template<typename PointT>
typename pcl::PointCloud<PointT>::Ptr ProcessPointClouds<PointT>::loadPcd(std::string file)
{

    typename pcl::PointCloud<PointT>::Ptr cloud (new pcl::PointCloud<PointT>);

    if (pcl::io::loadPCDFile<PointT> (file, *cloud) == -1) //* load the file
    {
        PCL_ERROR ("Couldn't read file \n");
    }
    std::cerr << "Loaded " << cloud->points.size () << " data points from "+file << std::endl;

    return cloud;
}


// Stream point cloud data files from given path
template<typename PointT>
std::vector<boost::filesystem::path> ProcessPointClouds<PointT>::streamPcd(std::string dataPath)
{

    std::vector<boost::filesystem::path> paths(boost::filesystem::directory_iterator{dataPath}, boost::filesystem::directory_iterator{});

    // sort files in accending order so playback is chronological
    sort(paths.begin(), paths.end());

    return paths;

}