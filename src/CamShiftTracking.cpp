/* written by wpf @ Singapore
 * 2013/5/2
 * based on the camshiftdemo.cpp in OpenCV distribution
 */

#include "endeffector_tracking/CamShiftTracking.h"

using namespace cv;

CamShiftTracking::CamShiftTracking()
:vmin(10)
,vmax(256)
,smin(30)
,hsize(16)
,phranges(hranges)
{
	hranges[0] = 0;
	hranges[1] = 180;
}

cv::Rect 
CamShiftTracking::getWindow(void)
{
	return trackWindow;
}

void 
CamShiftTracking::retrieveImage(const cv::Mat& img)
{
	this->frame = img.clone();
}

void 
CamShiftTracking::initTarget(const int *win_)
{
	selection.x 		= win_[1];
	selection.y 		= win_[0];
	selection.width 	= win_[3] - win_[1];
	selection.height 	= win_[2] - win_[0];

	int _vmin = vmin, _vmax = vmax;

	frame.copyTo(image);
	cvtColor(image, hsv, CV_BGR2HSV);
	inRange(hsv, Scalar(0, smin, MIN(_vmin,_vmax)),
			Scalar(180, 256, MAX(_vmin, _vmax)), mask);
	int ch[] = {0, 0};
	hue.create(hsv.size(), hsv.depth());
	mixChannels(&hsv, 1, &hue, 1, ch, 1);

	Mat roi(hue, selection), maskroi(mask, selection);
	calcHist(&roi, 1, 0, maskroi, hist, 1, &hsize, &phranges);
	normalize(hist, hist, 0, 255, CV_MINMAX);

	trackWindow = selection;
}

void
CamShiftTracking::track(void)
{
	frame.copyTo(image);

	cvtColor(image, hsv, CV_BGR2HSV);

	int _vmin = vmin, _vmax = vmax;

	inRange(hsv, Scalar(0, smin, MIN(_vmin,_vmax)),
			Scalar(180, 256, MAX(_vmin, _vmax)), mask);
	int ch[] = {0, 0};
	hue.create(hsv.size(), hsv.depth());
	mixChannels(&hsv, 1, &hue, 1, ch, 1);


	calcBackProject(&hue, 1, 0, hist, backproj, &phranges);
	backproj &= mask;
	RotatedRect trackBox = CamShift(backproj, trackWindow,
			TermCriteria( CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 10, 1 ));
	if( trackWindow.area() <= 1 )
	{
		int cols = backproj.cols, rows = backproj.rows, r = (MIN(cols, rows) + 5)/6;
		trackWindow = Rect(trackWindow.x - r, trackWindow.y - r,
				trackWindow.x + r, trackWindow.y + r) &
			Rect(0, 0, cols, rows);
	}

	ellipse( image, trackBox, Scalar(0,0,255), 3, CV_AA );

}

