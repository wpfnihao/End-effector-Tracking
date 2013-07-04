/**
 * @file mbtKltTracker.cpp
 * @brief The implementation of the mbtKltTracker class.
 * @author Pengfei Wu - wpfnihao@gmail.com
 * @version 1.0
 * @date 2013-05-29
 */

#include "endeffector_tracking/mbtKltTracker.h"

// opencv2 headers
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <conversions/image.h>


mbtKltTracker::mbtKltTracker()
:tracker()  // init the vpMbKltTracker
{
}

void
mbtKltTracker::initialize(std::string config_file, std::string model_name, std::string init_file)
{
	tracker.loadConfigFile(config_file);
	tracker.getCameraParameters(cam);
	tracker.setDisplayFeatures(true);
	tracker.setOgreVisibilityTest(true);
	tracker.loadModel(model_name);

	display.init(curImg);
	tracker.initClick(curImg, init_file);
	tracker.getPose(cMo);
}

bool 
mbtKltTracker::pubRst(sensor_msgs::Image msg, cv::Rect& box)
{
	tracker.display(curImg, cMo, cam, vpColor::red, 2);
	msg = visp_bridge::toSensorMsgsImage(curImg);
	return false;
}

void 
mbtKltTracker::retrieveImage(const sensor_msgs::ImageConstPtr& img)
{
	curImg = visp_bridge::toVispImage(*img);	
}

void 
mbtKltTracker::track(void)
{
	vpDisplay::display(curImg);
	tracker.track(curImg);
	tracker.getPose(cMo);
	tracker.display(curImg, cMo, cam, vpColor::red, 2);
	vpDisplay::displayFrame(curImg, cMo, cam, 0.025, vpColor::none, 3);
	vpDisplay::flush(curImg);
	vpTime::wait(40);
}
