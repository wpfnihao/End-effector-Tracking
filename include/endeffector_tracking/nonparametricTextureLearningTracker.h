/**
 * @file nonparametricTextureLearningTracker.h
 * @brief The header file of the non-parametric kernel density estimation based texture features learning tracker.
 * Try my best to make the tracker work and work well.
 * The method used here will be submitted to ICRA2014, no alternative.
 * @author Pengfei Wu - wpfnihao@gmail.com
 * @version 1.0
 * @date 2013-07-14
 */

#include "endeffector_tracking/cvBaseTracker.h"
#include "endeffector_tracking/cadModel.h"
#include "endeffector_tracking/textureTracker.h"

class nonparametricTextureLearningTracker: public cadModel, public cvBaseTracker
{
	public:
		/**
		 * @brief initialize the tracker during the initialization steps
		 */
		void init(const cv::Mat& img, vpHomogeneousMatrix cMo_, vpCameraParameters cam_, std::string init_file);

		/**
		 * @brief retrieve a frame of image for tracking
		 *
		 * @param img
		 */
		virtual inline void retrieveImage(const cv::Mat& img)
		{
			cv::cvtColor(img, curImg, CV_RGB2GRAY);
			this->processedImg = img.clone();
		}

		virtual void track(void);

		virtual bool pubRst(cv::Mat& img, cv::Rect& box);

	private:
		/**
		 * @brief the texture tracker, the texture based tracking procedure is done here.
		 */
		textureTracker texTracker;

		cv::Mat gradient;

	protected:
		void sobelGradient(const cv::Mat& curImg, cv::Mat& gradient);
};
