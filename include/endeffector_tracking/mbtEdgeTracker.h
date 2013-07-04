/**
 * @file mbtEdgeTracker.h
 * @brief The mbtEdgeTracker class wrap the vpMbEdgeTracker in the visp package
 * 			to incorporate with endeffector_tracking ROS package	
 * 			The moving edge tracker is used in this class
 * @author Pengfei Wu - wpfnihao@gmail.com
 * @version 1.0
 * @date 2013-05-28
 */

#include <visp/vpMbEdgeTracker.h>
#include <visp/vpImage.h>
#include <sensor_msgs/image_encodings.h>
#include <visp/vpCameraParameters.h>
#include <visp/vpPoseFeatures.h>
#include <image_transport/image_transport.h>

// for initialization
#include "visp/vpDisplay.h"
#include "visp/vpDisplayX.h"

class mbtEdgeTracker
{
	public:
		/**
		 * @brief the constructor of the class, the basic initialization steps are done here
		 */
		mbtEdgeTracker();

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

		// the interface for cooperating with other trackers
		// these two functions only needed to be called when this tracker is used to incorporate with others
		/**
		 * @brief get the pose from the upper level class, which usually provided by other trackers
		 *
		 * @param cMo_ the pose
		 */
		inline void getPose(vpHomogeneousMatrix& cMo_)
		{
			this->cMo = cMo_;
		}

		/**
		 * @brief publish the pose calculated by this tracker, which can be used by other trackers to refine the result
		 *
		 * @param cMo_ the pose
		 */
		inline void pubPose(vpHomogeneousMatrix& cMo_)
		{
			cMo_ = this->cMo;
		}

	protected:

	private:
		/**
		 * @brief the tracking procedure is actually done based on this class.
		 */
		vpMbEdgeTracker tracker;

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

		std::string cad_name;
};
