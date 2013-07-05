/**
 * @file track_ros.cpp
 * @brief The main tracking class
 * @author Pengfei Wu - wpfnihao@gmail.com
 * @version 1.0
 * @date 2013-05-23
 */

#include "ros/ros.h"
#include "endeffector_tracking/track_ros.h"

#include "visp/vpDisplay.h"
#include "visp/vpDisplayX.h"

// for conversion
#include "visp/vpMeterPixelConversion.h"


using namespace std;


/**
 * @brief 	constructor
 *			ROS procedure initialized here
 *
 * @param argc 		Number of the parameters from the command line
 * @param argv 		Pointer of the parameters from the command line
 */
CEndeffectorTracking::CEndeffectorTracking(int argc, char **argv)
:status(CEndeffectorTracking::INITIAL)
{

	//ros::NodeHandle n 
	//the node handle n is the private member variable of the class 

	//get param from the launch file or command line

	//param([param_name], [member_variable], [default_value])
	
	//FIXME:the param namespace is weird, how to figure out the namespace of an param?
	n.param<std::string>("endeffector_tracking/camera_topic", camera_topic, "camera/image_raw");
	//the config_file containing the info of the cam
	n.param<std::string>("endeffector_tracking/config_file", config_file, "config.xml");
	n.param<std::string>("endeffector_tracking/model_name", model_name, "config.cao");
	n.param<std::string>("endeffector_tracking/init_file", init_file, "config.init");


	image_transport::ImageTransport it(n);
	// subCam is member variable subscribe to the msg published by the camera
	// trackCallback do the tracking procedure
	// the callback function is a member function of a class, so we should point out which class it belongs to explicitly
	subCam = it.subscribe(camera_topic, 1, &CEndeffectorTracking::trackCallback, this);

	// publish message
	imgPub = it.advertise("hough", 1);
}

/**
 * @brief  	The tracking procedure is done here
 * 			Note that different trackers can be employed here
 * 			This function is only the glue function of ROS and all the other 
 * 			trackers
 * 			Only some very basic info. obtained from the ROS topic is processed here
 * 			INITIAL: the init points from .init file are obtained
 * 					 if use the houghLineBasedTracker, the model based on the init points are generated and maintained in the houghLineBasedTracker (not in this interface class).
 * 			LOST: this is the reinitialization step, including the initial pose of the target, corresponding tracking window, etc.
 * 			TRACKING: the tracking function provided by other tracking classes, and the tracked object is published as a ROS topic.
 *
 *
 * @param srcImg 	the sensor_msgs of ROS topic, 
 * 					the message will be transfered to OpenCV image and then stored.
 */
void 
CEndeffectorTracking::trackCallback(const sensor_msgs::ImageConstPtr& srcImg)
{
	// transform the image msg into cv::Mat and store in the member variable
	msg2mat(srcImg, this->curImg);

	//tracking based on different status
	if (status == CEndeffectorTracking::LOST || status == CEndeffectorTracking::INITIAL)
	{
		// init the model from file
		// only requires when initializing
		if (status == CEndeffectorTracking::INITIAL)
		{
			// get some basic info about the video
			cv::Mat grayImg;
			cv::cvtColor(curImg, grayImg, CV_BGR2GRAY);
			rows = grayImg.rows;
			cols = grayImg.cols;
			
			// parse the .init file, and get the init points
			getInitPoints();
			// get the info of the cube from the init points
			//hlTracker.initModel(initP);
			fbTracker.initModel(initP);
			fbTracker.init(curImg);

			meTracker.retrieveImage(srcImg);
			meTracker.initialize(config_file, model_name, init_file);
			//kltTracker.retrieveImage(srcImg);
			//kltTracker.initialize(config_file, model_name, init_file);
			//mekltTracker.retrieveImage(srcImg);
			//mekltTracker.initialize(config_file, model_name, init_file);
		}

		initializeTracker(srcImg);
		//hlTracker.retrieveImage(curImg);
		//hlTracker.initialize(this->cam, cMo, poseVector, rows, cols);
		fbTracker.retrieveImage(curImg);
		fbTracker.initialize(this->cam, cMo, poseVector, rows, cols);

		//csTracker.retrieveImage(curImg);
		//kbTracker.retrieveImage(curImg);
		//genTrackingWindow();
		//kbTracker.initTarget(window);
		//csTracker.initTarget(window);
		//mfTracker.init(window, curImg);

		//finish the initialization step and start to track
		status = CEndeffectorTracking::TRACKING;
	}	
	else if (status == CEndeffectorTracking::TRACKING)
	{
		//kbTracker.retrieveImage(curImg);
		//kbTracker.track();
		//csTracker.retrieveImage(curImg);
		//csTracker.track();
		//mfTracker.retrieveImage(curImg);
		//mfTracker.track();
		//this->kbtWindow = kbTracker.getWindow();
		//this->kbtWindow = mfTracker.getWindow();
		// temporarily used to draw object tracked by kbTracker
		//processedImg = curImg.clone();
		//cv::rectangle(processedImg, kbtWindow, cv::Scalar(0), 2);		


		//hlTracker.retrieveImage(curImg);
		//hlTracker.track();
		//if(hlTracker.pubRst(this->processedImg, this->TrackerWindow))
		//	status = CEndeffectorTracking::LOST;
		fbTracker.getPose(cMo);
		fbTracker.retrieveImage(curImg);
		for (int i = 0; i < 1; i++)
		{
		fbTracker.track();
		fbTracker.refinePose();
		}
		fbTracker.pubPose(cMo);
		if(fbTracker.pubRst(this->processedImg, this->TrackerWindow))
			status = CEndeffectorTracking::LOST;
		
		//meTracker.getPose(cMo);
		//meTracker.retrieveImage(srcImg);
		//meTracker.track();
		//meTracker.pubPose(cMo);
		//kltTracker.retrieveImage(srcImg);
		//kltTracker.track();
		//mekltTracker.retrieveImage(srcImg);
		//mekltTracker.track();
		// sensor_msgs::Image msg;	
		// meTracker.pubRst(msg, TrackerWindow);
		// imgPub.publish(msg);

		pubRst(srcImg);
	}
}

/**
 * @brief convert from the ros image message to a CvImage
 *
 * @param srcImg
 * @param curImg_
 */
void 
CEndeffectorTracking::msg2mat(const sensor_msgs::ImageConstPtr& srcImg, cv::Mat& curImg_)	
{
	cv_bridge::CvImagePtr cv_ptr;
	try
	{
		cv_ptr = cv_bridge::toCvCopy(srcImg, sensor_msgs::image_encodings::BGR8);
	}
	catch (cv_bridge::Exception& e)
	{
		ROS_ERROR("%s",e.what());
		return;
	}	

	curImg_ = cv_ptr->image; 

}


/**
 * @brief 			use the vpMbEdgeTracker to initialize the tracker
 *
 * @param srcImg 	the sensor_msgs of ROS topic, it will be transfered to vpImage in the function
 */
void 
CEndeffectorTracking::initializeTracker(const sensor_msgs::ImageConstPtr& srcImg)
{
	// the vpImg is a mono channel image
	vpImage<unsigned char> vpImg;
	// the operator:=() will allocate the memory automatically
	vpImg = visp_bridge::toVispImage(*srcImg);

	// We only use the tracker in visp to initialize the program
	vpMbEdgeTracker initializer;
	// NOTE:the params can also be retrieved from the sensor_msgs::CameraInfo using the visp_bridge package
	initializer.loadConfigFile(config_file);		
	initializer.getCameraParameters(cam);
	initializer.loadModel(model_name);
	initializer.setCovarianceComputation(true);

	//initial the display
	//these steps are not shown in the manual document
	vpDisplayX display;
	display.init(vpImg);

	initializer.initClick(vpImg, init_file);
	// obtain the initial pose of the model
	cMo = initializer.getPose();
	poseVector.buildFrom(cMo);

	//clean up
	vpXmlParser::cleanup();
}


/**
 * @brief  	Init the model from the *.init file, only used for the cube tracking
 * 			initClick in the ViSP software is referred for the IO operations used here
 *  		fstream is used here
 */
void 
CEndeffectorTracking::getInitPoints(void)
{
	std::fstream finit;	
	finit.open(init_file.c_str(), std::ios::in);
	if (finit.fail())
	{
		std::cout << "cannot read " << init_file << std::endl;
		throw vpException(vpException::ioError, "cannot read init file for model initialization!");
	}

	//file parser
	//number of points
	//X Y Z
	//X Y Z
	 
	double X,Y,Z ;

	unsigned int n ;
	finit >> n ;
	std::cout << "number of points " << n << std::endl ;
	for (unsigned int i=0 ; i < n ; i++)
	{
		finit >> X ;
		finit >> Y ;
		finit >> Z ;
		// NOTE: X,Y,Z are small float variables in meter, do NOT use int to save them
		cv::Point3f curP(X, Y, Z);
		// initialize the initP, which is the member variable of the class
		initP.push_back(curP); // (X,Y,Z)
	}

	finit.close();
}

/**
 * @brief publish the processed Img or the tracking rst
 * 			TODO: modifications required.
 *
 * @param srcImg
 */
void
CEndeffectorTracking::pubRst(const sensor_msgs::ImageConstPtr& srcImg)
{
	// Step 5: publish the tracking result

	cv_bridge::CvImage out_msg;
	out_msg.header = srcImg->header;
	out_msg.encoding = sensor_msgs::image_encodings::BGR8;
	out_msg.image = processedImg;

	imgPub.publish(out_msg.toImageMsg());
}
