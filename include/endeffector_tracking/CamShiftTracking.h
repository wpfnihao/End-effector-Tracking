/* written by wpf @ Singapore
 * 2013/5/2
 * based on the camshiftdemo.cpp in OpenCV distribution
 */

#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

class CamShiftTracking
{
	public:
		CamShiftTracking();
		void retrieveImage(const cv::Mat& img);
		void track(void);
		void initTarget(const int *win_);
		cv::Rect getWindow(void);
	private:
		cv::Mat image, frame;

		cv::Point origin;
		cv::Rect selection;
		int vmin, vmax, smin;
		cv::Rect trackWindow;
		cv::RotatedRect trackBox;
		int hsize;
		float hranges[2];
		const float* phranges;
		cv::Mat hsv, hue, mask, hist, backproj;
	protected:
};
