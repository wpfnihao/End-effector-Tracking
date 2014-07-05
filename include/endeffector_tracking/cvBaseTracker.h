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

//#include <conversions/image.h>
#include <visp_bridge/image.h>

class cvBaseTracker
{
	protected:
		/**
		 * @brief  	the current pose
		 */
		vpHomogeneousMatrix cMo;
		vpHomogeneousMatrix p_cMo;

		vpCameraParameters cam;

	/**
	 * @brief current cvImg maintained by the class
	 * 			TODO: the maintaining of the img
	 */
		cv::Mat curImg, preImg;
		cv::Mat processedImg;
	public:
		/**
		 * @brief 		retrieve Image from the up level class
		 *
		 * @param img 	the input image
		 */
		virtual void retrieveImage(const cv::Mat& img) = 0;

		/**
		 * @brief  tracking is done here
		 * 			this is a pure virtual function, and should be implemented in the derived classes.
		 */
		virtual void track(void) = 0;

		virtual bool pubRst(cv::Mat& img, cv::Rect& box) = 0;

		/**
		 * @brief WARNING: note here that the getPose function copy the cMo_ into the p_cMo
		 * not the cMo
		 *
		 * @param cMo_
		 */
		inline void getPose(vpHomogeneousMatrix& cMo_)
		{
			this->p_cMo = cMo_;
		}

		inline void getPoseCur(vpHomogeneousMatrix& cMo_)
		{
			this->cMo = cMo_;
		}

		inline void pubPose(vpHomogeneousMatrix& cMo_)
		{
			cMo_ = this->cMo;
		}
};

#endif
