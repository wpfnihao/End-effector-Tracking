/* This class is written by wpf @ Singapore
 * 2013/5/9
 */
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/video/tracking.hpp"
#include <iostream>
#include <math.h>

class medianFlowTracking
{
	public:
		medianFlowTracking();
		void track(void);
		void init(const int *win_, const cv::Mat& img);
		void retrieveImage(const cv::Mat& img);
		cv::Rect getWindow(void);
	private:
		// previous image and current image
		cv::Mat pImg, cImg;
		std::vector<cv::Mat> pPyramid, cPyramid;

		// motion detected
		float dx, dy;

		// tracking box
		cv::Rect pbox, cbox;

		// features can be tracked stably
		std::vector<cv::Point2f> pFeatures, cFeatures;

		// multi-scale
		int maxLevel;

		// window size for optical flow calc and features initialization
		int winSize;
	protected:
		// generate initial features for tracking
		void genInitFeatures(cv::Rect box, int size, std::vector<cv::Point2f>& features);

		// find the good features through forward-backward stability
		void findStableFeatures(
				const std::vector<cv::Point2f>& 		features1, 
				const std::vector<cv::Point2f>& 		features2, 
				std::vector<cv::Point2f>& 			cFeatures_, 
				const std::vector<unsigned char>& 	status1,
				const std::vector<unsigned char>& 	status2,
				std::vector<cv::Point2f>& 			outputFeatures,
				std::vector<cv::Point2f>& 			motion_,
				float 								th
				);

		// find motion of the tracking box from the motion vector
		void findMotion(const std::vector<cv::Point2f>& motion_, float& dx_, float& dy_);
};
