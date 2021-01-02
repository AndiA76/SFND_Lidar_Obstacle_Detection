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

/* Visualization of a 3D Lidar environment scan of driving scenarios on highway
(simuation) or in an urban environment (replay) */

#include "sensors/lidar.h"
#include "render/render.h"
#include "processPointClouds.h"
// using templates for processPointClouds so also include .cpp to help linker
#include "processPointClouds.cpp"
#include "cluster3D.cpp"


// Init simple 3d highway environment using PCL for exploring self-driving car Lidar sensors
std::vector<Car> initHighway(bool render_scene, pcl::visualization::PCLVisualizer::Ptr& viewer)
{
    // Create positions of the ego vehicle and other cars (target objects)
    Car egoCar( Vect3(0,0,0), Vect3(4,2,2), Color(0,1,0), "egoCar");
    Car car1( Vect3(15,0,0), Vect3(4,2,2), Color(0,0,1), "car1");
    Car car2( Vect3(8,-4,0), Vect3(4,2,2), Color(0,0,1), "car2");	
    Car car3( Vect3(-12,4,0), Vect3(4,2,2), Color(0,0,1), "car3");
  
    // Push the car locations to the output vector containing all car locations
    std::vector<Car> cars;
    cars.push_back(egoCar);
    cars.push_back(car1);
    cars.push_back(car2);
    cars.push_back(car3);

    // Render the 3D scene with ego vehicle and other cars (target objects) if activated
    if(render_scene)
    {
        renderHighway(viewer);
        egoCar.render(viewer);
        car1.render(viewer);
        car2.render(viewer);
        car3.render(viewer);
    }

    return cars;
}


// Visualize 3D Lidar scan of our simple highway scenario uisng simulation
void simpleHighway(pcl::visualization::PCLVisualizer::Ptr& viewer)
{
    // ----------------------------------------------------
    // -----Open 3D viewer and display simple highway -----
    // ----------------------------------------------------
    
    // RENDERING OPTIONS
    bool render_scene = false; // render scene with (true) or without (false) target objects
    bool render_point_cloud = false;
    bool render_rays = false;
    bool render_obstacles = true;
    bool render_plane = true;
    bool render_clusters = true;
    bool render_boxes = true;

    std::vector<Car> cars = initHighway(render_scene, viewer);
    
    // Create lidar sensor
    // Define slope of the driving ground (even ground => 0.0)
    float groundSlope = 0.0;

    // Create lidar sensor on the heap using "new" keyword
    // Memory on stack is limited to ~2 MB => more memory on heap
    Lidar* lidar = new Lidar(cars, groundSlope);

    // Scan the environment and get a point cloud
    pcl::PointCloud<pcl::PointXYZ>::Ptr inputCloud = lidar->scan();

    // Visualize lidar scan (lidar rays and original point cloud without segmentation or obstacle clustering)
    if (render_rays)
        renderRays(viewer, lidar->position, inputCloud);
    if (render_point_cloud)
        renderPointCloud(viewer, inputCloud, "inputCloud", Color(1, 1, 1));

    // Create point processor
    //ProcessPointClouds<pcl::PointXYZ> pointProcessor;  // on stack
    ProcessPointClouds<pcl::PointXYZ>* pointProcessor = new ProcessPointClouds<pcl::PointXYZ>();  // on heap

    // Get number of points in input cloud
    std::cout << "Total number of points in input cloud:" << std::endl;
    pointProcessor->numPoints(inputCloud);

    // Number of iterations for plane segmentation
    int numIter = 100;
    //int numIter = 50;
    //int numIter = 25;
    // Distance tolerance thresholds in [m] to find inliers of the driving ground plane
    //float distTol = 0.5;
    //float distTol = 0.3;
    float distTol = 0.2;

    // Plane segmentation => road (plane) and obstacles => using a given number of iterations and a distance tolerance threshold
    //std::pair<pcl::PointCloud<pcl::PointXYZ>::Ptr, pcl::PointCloud<pcl::PointXYZ>::Ptr> segmentedCloud = pointProcessor.SegmentPlane(inputCloud, numIter, distTol);  // on stack
    //std::pair<pcl::PointCloud<pcl::PointXYZ>::Ptr, pcl::PointCloud<pcl::PointXYZ>::Ptr> segmentedCloud = pointProcessor->SegmentPlane(inputCloud, numIter, distTol);  // on heap
    std::pair<pcl::PointCloud<pcl::PointXYZ>::Ptr, pcl::PointCloud<pcl::PointXYZ>::Ptr> segmentedCloud = pointProcessor->SegmentPlaneRansac3D_v1(inputCloud, numIter, distTol);  // on heap
    //std::pair<pcl::PointCloud<pcl::PointXYZ>::Ptr, pcl::PointCloud<pcl::PointXYZ>::Ptr> segmentedCloud = pointProcessor->SegmentPlaneRansac3D_v2(inputCloud, numIter, distTol);  // on heap

    // Visualize lidar scan with separated point clouds for road (plane) and obstacles in different colors
    if (render_obstacles)
        renderPointCloud(viewer, segmentedCloud.first, "obstacleCloud", Color(1, 0, 0));
    if (render_plane)
        renderPointCloud(viewer, segmentedCloud.second, "planeCloud", Color(0, 1, 0));

    // Define search parameters to find clusters
    // Set distance tolerance in [m] to separate obstacle clusters from one another (e. g. two cars)
    //const float clusterTolerance = 2.0;
    const float clusterTolerance = 1.0;
    //const float clusterTolerance = 0.5;
    // Set minimum number of agglomerated points to form a target object cluster (e. g. another car)
    const int minClusterSize = 3;
    // Set maximum number of agglomerated points to form a target object cluster (e. g. another car)
    const int maxClusterSize = 30;
    // Cluster point cloud points above the drivable plane into separated obstacle clusters
    //std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> cloudClusters = pointProcessor->Clustering(segmentedCloud.first, clusterTolerance, minClusterSize, maxClusterSize);
    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> cloudClusters = pointProcessor->ClusteringKdTree3D(segmentedCloud.first, clusterTolerance, minClusterSize, maxClusterSize);

    // Visualize the obstacle clusters in different colors
    int clusterId = 0;
    // Define a set of colors (here: red, yellow, blue)
    std::vector<Color> colors = {Color(1,0,0), Color(1,1,0), Color(0,0,1)};

    // Iterate through the vector of point clouds
    for(pcl::PointCloud<pcl::PointXYZ>::Ptr cluster : cloudClusters)
    {
        if (render_clusters)
        {
            // Render point obstacle point clouds for each cluster id
            std::cout << "cluster size ";
            pointProcessor->numPoints(cluster);
            renderPointCloud(viewer,cluster,"obstacleCloud"+std::to_string(clusterId),colors[clusterId]);
        }
        if (render_boxes)
        {
            // Render a bounding box around each (unique) cluster id
            Box box = pointProcessor->axisAlignedBoundingBox(cluster);
            //BoxQ box = pointProcessor->orientedBoundingBox(cluster);  // PCA is a sub-optimale strategy to fit 3D boxes around vehicles
            renderBox(viewer,box,clusterId);
        }
         // Increment cluster id
        ++clusterId;
    }
  
}


// Visualize 3D Lidar scan of city scenarios using real data recordings
void cityBlock(pcl::visualization::PCLVisualizer::Ptr& viewer, ProcessPointClouds<pcl::PointXYZI>* pointProcessor, pcl::PointCloud<pcl::PointXYZI>::Ptr inputCloud)
{
    // ----------------------------------------------------
    //       Open 3D viewer and display City Block
    // ----------------------------------------------------
    
    // RENDERING OPTIONS
    bool render_input_cloud = false;
    bool render_voxel_grid_cloud = true;
    bool render_plane = true;
    bool render_obstacles = false;
    bool render_obstacle_clusters = true;
    bool render_bounding_boxes = true;

    // Get number of points in input cloud before downsampling
    std::cout << "Total number of points in input cloud before voxel grid filtering:" << std::endl;
    pointProcessor->numPoints(inputCloud);

    // Render original point cloud data before downsampling
    if (render_input_cloud)
        renderPointCloud(viewer,inputCloud,"inputCloud",Color(1, 1, 1));

    // Set downsampling filter resolution
    //float filterRes = 0.5;
    float filterRes = 0.3;
    //float filterRes = 0.2;
    //float filterRes = 0.15;

    // Define coordinates of crop box corner points to pick the space to be rendered from the 3D Lidar scan
    /* default: 
    float xmin_crop = -10.0;
    float ymin_crop = -5.0;
    float zmin_crop = -2.0;
    float xmax_crop = 30.0;
    float ymax_crop = 8.0;
    float zmax_crop = 1.0;
    */
    /* test: */
    float xmin_crop = -10.0;
    float ymin_crop = -6.0;
    float zmin_crop = -2.0;
    float xmax_crop = 30.0;
    float ymax_crop = 7.0;
    float zmax_crop = 2.0;
    
    // Initialize quaternion representation of crop box corner points (q = 1)
    Eigen::Vector4f minPointQ (xmin_crop, ymin_crop, zmin_crop, 1);
    Eigen::Vector4f maxPointQ (xmax_crop, ymax_crop, zmax_crop, 1);
    //Eigen::Vector4f minPointQ = Eigen::Vector4f(-10, -5, -2, 1);
    //Eigen::Vector4f maxPointQ = Eigen::Vector4f(30, 8, 1, 1);

    // Downsample poind cloud data using voxel grid approach
    pcl::PointCloud<pcl::PointXYZI>::Ptr voxelGridCloud = pointProcessor->FilterCloud(inputCloud,filterRes,minPointQ,maxPointQ);

    // Get number of points in input cloud after downsampling
    std::cout << "Total number of points in voxel grid cloud after downsampling and ROI cropping:" << std::endl;
    pointProcessor->numPoints(voxelGridCloud);

    // Render the point cloud data of the 3D Lidar scan of the city scene
    if (render_voxel_grid_cloud)
        renderPointCloud(viewer,voxelGridCloud,"voxelGridCloud",Color(0, 0.5, 0));
    
    // Number of iterations for plane segmentation
    int numIter = 100;
    //int numIter = 50;
    //int numIter = 25;
    // Distance tolerance thresholds in [m] to find inliers of the driving ground plane
    //float distTol = 0.5;
    //float distTol = 0.3;
    float distTol = 0.2;

    // Plane segmentation separating road (plane) from obstacles => using given number of iterations and a distance tolerance threshold
    //std::pair<pcl::PointCloud<pcl::PointXYZI>::Ptr, pcl::PointCloud<pcl::PointXYZI>::Ptr> segmentedCloud = pointProcessor->SegmentPlane(voxelGridCloud, numIter, distTol);  // on heap
    std::pair<pcl::PointCloud<pcl::PointXYZI>::Ptr, pcl::PointCloud<pcl::PointXYZI>::Ptr> segmentedCloud = pointProcessor->SegmentPlaneRansac3D_v1(voxelGridCloud, numIter, distTol);  // on heap
	//std::pair<pcl::PointCloud<pcl::PointXYZI>::Ptr, pcl::PointCloud<pcl::PointXYZI>::Ptr> segmentedCloud = pointProcessor->SegmentPlaneRansac3D_v2(voxelGridCloud, numIter, distTol);

    // Visualize lidar scan with segmented point clouds for road (plane) and obstacles in different colors
    if (render_obstacles)
        renderPointCloud(viewer, segmentedCloud.first, "obstacleCloud", Color(1, 0, 0));
    if (render_plane)
        renderPointCloud(viewer, segmentedCloud.second, "planeCloud", Color(0, 1, 0));
    
    // Define search parameters to find clusters
    // Set distance tolerance in [m] to separate obstacle clusters from one another (e. g. two cars)
    const float clusterTolerance = 0.53;
    // Set minimum number of agglomerated points to form a target object cluster (e. g. another car)
    const int minClusterSize = 10;
    // Set maximum number of agglomerated points to form a target object cluster (e. g. another car)
    const int maxClusterSize = 500;
    // Cluster target objects (obstacles)
    //std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> cloudClusters = pointProcessor->Clustering(segmentedCloud.first, clusterTolerance, minClusterSize, maxClusterSize);
    std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> cloudClusters = pointProcessor->ClusteringKdTree3D(segmentedCloud.first, clusterTolerance, minClusterSize, maxClusterSize);

    // Set object colors
    std::vector<Color> colors = {Color(1, 0, 0), Color(1, 1, 0), Color(0, 1, 1), Color(0, 0, 1), Color(1, 0, 1)};
    // Loop through all cluster objects and render them with different colors
    int clusterId = 0;
    for (pcl::PointCloud<pcl::PointXYZI>::Ptr cluster : cloudClusters)
    {
        // Render point cluster
        std::cout << "cluster size ";
        pointProcessor->numPoints(cluster);
        if (render_obstacle_clusters)
            renderPointCloud(viewer, cluster, "obstacleCluster"+std::to_string(clusterId), colors[clusterId%colors.size()]);

        // Render an axis aligned 3D bounding box around the point cluster
        Box box = pointProcessor->axisAlignedBoundingBox(cluster);
        // Render an PCA-oriented 3D bounding box around the point cluster => Not optimal for vehicle detection!
        //BoxQ box = pointProcessor->orientedBoundingBox(cluster);
        if (render_bounding_boxes)
            renderBox(viewer, box, clusterId);
        ++clusterId;
    }

}


//setAngle: SWITCH CAMERA ANGLE {XY, TopDown, Side, FPS}
void initCamera(CameraAngle setAngle, pcl::visualization::PCLVisualizer::Ptr& viewer)
{

    // Set background color (black)
    viewer->setBackgroundColor (0, 0, 0);
    
    // Set camera position and angle
    viewer->initCameraParameters();

    // Distance away in meters
    int distance = 16;
    
    // Set camera view angle
    switch(setAngle)
    {
        case XY : viewer->setCameraPosition(-distance, -distance, distance, 1, 1, 0); break;
        case TopDown : viewer->setCameraPosition(0, 0, distance, 1, 0, 1); break;
        case Side : viewer->setCameraPosition(0, -distance, 0, 0, 0, 1); break;
        case FPS : viewer->setCameraPosition(-10, 0, 0, 0, 0, 1);
    }

    if(setAngle!=FPS)
        viewer->addCoordinateSystem (1.0);
    
}


// Run the visualization of the 3D environment
int main (int argc, char** argv)
{
    std::cout << "starting enviroment" << std::endl;

    // Choose options: simple highway (0) single frame city block (1), frame sequence city block (2)
    uint option = 2;

    if (option == 0)
    {
        /* Simulation of simple highway scenario */

        // Initialize viewer and camera position
        pcl::visualization::PCLVisualizer::Ptr viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
        CameraAngle setAngle = XY;
        //CameraAngle setAngle = TopDown;
        //CameraAngle setAngle = Side;
        //CameraAngle setAngle = FPS;
        initCamera(setAngle, viewer);

        // Visualize the highway scenario
        simpleHighway(viewer);

        while (!viewer->wasStopped ())
        {
            viewer->spinOnce ();
        }
    }
    else if (option == 1)
    {
        /* Recorded city block single frame */

        // Initialize viewer and camera position
        pcl::visualization::PCLVisualizer::Ptr viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
        CameraAngle setAngle = XY;
        //CameraAngle setAngle = TopDown;
        //CameraAngle setAngle = Side;
        //CameraAngle setAngle = FPS;
        initCamera(setAngle, viewer);
        
        // Create point cloud processor (on heap) for 3D Lidar points with intensity values
        ProcessPointClouds<pcl::PointXYZI>* pointProcessorI = new ProcessPointClouds<pcl::PointXYZI>();

        // Load point cloud data from file (x-, y-, z-coordinates plus intensity values)
        pcl::PointCloud<pcl::PointXYZI>::Ptr inputCloudI = pointProcessorI->loadPcd("../src/sensors/data/pcd/data_1/0000000000.pcd");
        //pcl::PointCloud<pcl::PointXYZI>::Ptr inputCloudI = pointProcessorI->loadPcd("../src/sensors/data/pcd/data_2/0000000000.pcd");

        // Visualize the point cloud scan of the city block scene
        cityBlock(viewer, pointProcessorI, inputCloudI);

        while (!viewer->wasStopped ())
        {
            viewer->spinOnce ();
        }
    }
    else
    {
        /* Recorded city block frame sequence */

        // Initialize viewer and camera position
        pcl::visualization::PCLVisualizer::Ptr viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
        //CameraAngle setAngle = XY;
        //CameraAngle setAngle = TopDown;
        //CameraAngle setAngle = Side;
        CameraAngle setAngle = FPS;
        initCamera(setAngle, viewer);

        // Create point cloud processor (on heap) for 3D Lidar points with intensity values
        ProcessPointClouds<pcl::PointXYZI>* pointProcessor = new ProcessPointClouds<pcl::PointXYZI>();

        // Init input point cloud
        pcl::PointCloud<pcl::PointXYZI>::Ptr inputCloud;

        // Init data stream iterator
        std::vector<boost::filesystem::path> stream = pointProcessor->streamPcd("../src/sensors/data/pcd/data_1");
        //std::vector<boost::filesystem::path> stream = pointProcessor->streamPcd("../src/sensors/data/pcd/data_2");
        auto streamIterator = stream.begin();

        while (!viewer->wasStopped ())
        {
            // Clear viewer
            viewer->removeAllPointClouds();
            viewer->removeAllShapes();

            // Load pcd and run obstacle detection process
            inputCloud = pointProcessor->loadPcd((*streamIterator).string());

            // Framewise visualize the point cloud scans of the city block scene
            cityBlock(viewer, pointProcessor, inputCloud);

            streamIterator++;
            if (streamIterator == stream.end())
                streamIterator = stream.begin();

            viewer->spinOnce ();
        }
    }
    
}