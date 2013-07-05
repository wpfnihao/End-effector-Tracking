/**
 * @file mbtEdgeKltTracker.h
 * @brief the wrap of the vpMbEdgeKltTracker class
 * @author Pengfei Wu - wpfnihao@gmail.com
 * @version 1.0
 * @date 2013-05-29
 */

// for initialization
#include "visp/vpDisplay.h"
#include "visp/vpDisplayX.h"

// for baseTracker
#include "endeffector_tracking/baseTracker.h"

class mbtEdgeKltTracker: public baseTracker
{
	public:
		/**
		 * @brief the constructor of the class, the basic initialization steps are done here
		 */
		mbtEdgeKltTracker();

		/**
		 * @brief initialize the tracker
		 */
		virtual void initialize(std::string config_file, std::string model_name, std::string init_file);

		/**
		 * @brief publish the tracked rst
		 *
		 * @param img
		 * @param box
		 *
		 * @return 
		 */
		virtual bool pubRst(sensor_msgs::Image msg, cv::Rect& box);

		/**
		 * @brief  tracking is done here
		 */
		virtual void track(void);

	protected:

	private:
		/**
		 * @brief the tracking procedure is actually done based on this class.
		 */
		vpMbEdgeKltTracker tracker;
};
