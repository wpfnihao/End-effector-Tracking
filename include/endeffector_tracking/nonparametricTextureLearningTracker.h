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

class textureTracker: public cadModel, public cvBaseTracker
{
	public:
		void init();
		virtual void retrieveImage(const cv::Mat& img);
		virtual void track(void);
		virtual bool pubRst(cv::Mat& img, cv::Rect& box);
	private:
	protected:
}
