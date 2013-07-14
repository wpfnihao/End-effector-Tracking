/**
 * @file cvBaseTracker.h
 * @brief The base class used for opencv data structure based trackers.
 * @author Pengfei Wu - wpfnihao@gmail.com
 * @version 1.0
 * @date 2013-07-14
 */

#ifndef CVBASETRACKER_INCLUDED
#define CVBASETRACKER_INCLUDED


#include <visp/vpMbEdgeTracker.h>
#include <visp/vpImage.h>
#include <sensor_msgs/image_encodings.h>
#include <visp/vpCameraParameters.h>
#include <visp/vpPoseFeatures.h>
#include <image_transport/image_transport.h>

#include <conversions/image.h>

class cvBaseTracker
{
	protected:
		/**
		 * @brief  	the current pose
		 */
		vpHomogeneousMatrix cMo;

		vpCameraParameters cam;

	public:
		/**
		 * @brief 		retrieve Image from the up level class
		 *
		 * @param img 	the input image
		 */
		void retrieveImage(const cv::Mat& img);

		/**
		 * @brief  tracking is done here
		 * 			this is a pure virtual function, and should be implemented in the derived classes.
		 */
		virtual void track(void) = 0;

		/**
		 * @brief 		retrieve Image from the up level class
		 *
		 * @param img 	the input image
		 */
		virtual void retrieveImage(const cv::Mat& img) = 0;

		virtual bool pubRst(cv::Mat& img, cv::Rect& box) = 0;

		inline void getPose(vpHomogeneousMatrix& cMo_)
		{
			this->p_cMo = cMo_;
		}

		inline void pubPose(vpHomogeneousMatrix& cMo_)
		{
			cMo_ = this->cMo;
		}
}
#endif
