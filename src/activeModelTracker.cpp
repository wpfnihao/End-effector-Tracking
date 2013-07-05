/**
 * @file activeModelTracker.cpp
 * @brief The active model based tracker. The shape prior is obtained from the 
 * 			3d CAD model.
 * @author Pengfei Wu - wpfnihao@gmail.com
 * @version 1.0
 * @date 2013-07-05
 */

// user defined
#include "endeffector_tracking/activeModelTracker.h"

void
activeModelTracker::track(void)
{
}

void 
activeModelTracker::deformSilhouette(vpHomogeneousMatrix& cMo_)
{
	// step 1: based on the pose and cad model, forward project the silhouette of the model onto the image plane.
	
	//
	// step 2: actively find the high gradient region
	//
	// step 3: based on the tentatively moved control points, estimate the pose
}
