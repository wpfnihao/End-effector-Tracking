/**
 * @file mbtEdgeTracker.h
 * @brief The mbtEdgeTracker class wrap the vpMbEdgeTracker in the visp package
 * 			to incorporate with endeffector_tracking ROS package	
 * 			The moving edge tracker is used in this class
 * @author Pengfei Wu - wpfnihao@gmail.com
 * @version 1.0
 * @date 2013-05-28
 */

// for initialization
#include "visp/vpDisplay.h"
#include "visp/vpDisplayX.h"

// for baseTracker
#include "endeffector_tracking/baseTracker.h"

class mbtEdgeTracker: public baseTracker
{
	public:
		/**
		 * @brief the constructor of the class, the basic initialization steps are done here
		 */
		mbtEdgeTracker();

		/**
		 * @brief initialize the tracker
		 * this is a pure virtual function, must be implemented in any tracker derived from this base class
		 */
		virtual void initialize(std::string config_file, std::string model_name, std::string init_file);

		/**
		 * @brief publish the tracked rst
		 *
		 * @param msg
		 * @param box
		 *
		 * @return 
		 */
		virtual bool pubRst(sensor_msgs::Image msg, cv::Rect& box);

		/**
		 * @brief  tracking is done here
		 * a pure virtual function
		 */
		virtual void track(void);

	protected:

	private:
		/**
		 * @brief the tracking procedure is actually done based on this class.
		 */
		vpMbEdgeTracker tracker;
};
