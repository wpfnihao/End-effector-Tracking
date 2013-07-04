/**
 * @file mbtEdgeKltTracker.h
 * @brief the wrap of the vpMbEdgeKltTracker class
 * @author Pengfei Wu - wpfnihao@gmail.com
 * @version 1.0
 * @date 2013-05-29
 */

#include <visp/vpMbEdgeKltTracker.h>
#include <visp/vpImage.h>
#include <sensor_msgs/image_encodings.h>
#include <visp/vpCameraParameters.h>
#include <visp/vpPoseFeatures.h>
#include <image_transport/image_transport.h>

// for initialization
#include "visp/vpDisplay.h"
#include "visp/vpDisplayX.h"

class mbtEdgeKltTracker
{
	public:
		/**
		 * @brief the constructor of the class, the basic initialization steps are done here
		 */
		mbtEdgeKltTracker();

		/**
		 * @brief initialize the tracker
		 */
		void initialize(std::string config_file, std::string model_name, std::string init_file);

		/**
		 * @brief publish the tracked rst
		 *
		 * @param img
		 * @param box
		 *
		 * @return 
		 */
		bool pubRst(sensor_msgs::Image msg, cv::Rect& box);

		/**
		 * @brief retrieve image from the up level class
		 *
		 * @param img
		 */
		void retrieveImage(const sensor_msgs::ImageConstPtr& img);

		/**
		 * @brief  tracking is done here
		 */
		void track(void);

	protected:

	private:
		/**
		 * @brief the tracking procedure is actually done based on this class.
		 */
		vpMbEdgeKltTracker tracker;

		/**
		 * @brief current image
		 */
		vpImage<unsigned char> curImg;

		/**
		 * @brief current camera pose
		 */
		vpHomogeneousMatrix cMo;

		/**
		 * @brief camera parameter
		 */
		vpCameraParameters cam;

		/**
		 * @brief for display the result
		 */
		vpDisplayX display;
};
