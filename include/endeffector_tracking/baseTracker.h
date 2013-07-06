/**
 * @file baseTracker.h
 * @brief the virtual base class of all the model based tracker
 * 			This class provides the common interface of all the trackers,
 * 			such as pubRst, track, retrieveImage, getPose, pubPose, initialize.
 * @author Pengfei Wu - wpfnihao@gmail.com
 * @version 1.0
 * @date 2013-07-05
 */

#ifndef BASETRACKER_INCLUDED
#define BASETRACKER_INCLUDED

#include <visp/vpMbEdgeTracker.h>
#include <visp/vpImage.h>
#include <sensor_msgs/image_encodings.h>
#include <visp/vpCameraParameters.h>
#include <visp/vpPoseFeatures.h>
#include <image_transport/image_transport.h>

#include <conversions/image.h>
class baseTracker
{
	protected:

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

	public:
		/**
		 * @brief initialize the tracker
		 * this is a pure virtual function, must be implemented in any tracker derived from this base class
		 */
		virtual void initialize(std::string config_file, std::string model_name, std::string init_file) = 0;

		/**
		 * @brief publish the tracked rst
		 *
		 * @param msg
		 * @param box
		 *
		 * @return 
		 */
		virtual bool pubRst(sensor_msgs::Image msg, cv::Rect& box) = 0;

		/**
		 * @brief retrieve image from the up level class
		 *
		 * @param img
		 */
		inline void retrieveImage(const sensor_msgs::ImageConstPtr& img)
		{
			curImg = visp_bridge::toVispImage(*img);	
		}

		/**
		 * @brief  tracking is done here
		 * a pure virtual function
		 */
		virtual void track(void) = 0;

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
};

#endif
