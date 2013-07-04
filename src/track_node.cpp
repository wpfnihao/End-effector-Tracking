/**
 * @file track_node.cpp
 * @brief The program entry of the endeffector_tracking system
 * @author Pengfei Wu
 * @version 1.0
 * @date 2013-03-20
 */

#include <ros/ros.h>



// user defined
#include "endeffector_tracking/track_ros.h"


/**
 * @brief The program entry point 
 *
 * @param argc
 * @param argv
 *
 * @return 
 */
int main(int argc, char **argv)
{
	//instantiate the endeffector_tracking class
	//the ros initial steps are done in the constructor, so only instantiate the CEndeffectorTracking class is sufficient
	std::string name = "endeffector_tracking";
	ros::init(argc, argv, name);
	CEndeffectorTracking e2t(argc, argv);	
	ros::spin();
	return 0;
}
