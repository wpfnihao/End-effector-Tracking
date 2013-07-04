/* This class is written by wpf @ Singapore
 * 2013/4/22
 *
 * based on the source code of Gurpinder Singh Sandhu (with the permission of the License Agreement)
 * The original License Agreement is listed below:
 */
/*IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.

  By downloading, copying, installing or using the software you agree to this license.
  If you do not agree to this license, do not download, install,
  copy or use the software.


  License Agreement

  Copyright (C) 2013, Gurpinder Singh Sandhu, all rights reserved.

  Redistribution and use in source and binary forms, with or without modification,
  are permitted provided that the following conditions are met:

 * Redistributions of source code must retain the above copyright notice,
 this list of conditions and the following disclaimer.

 * Redistributions in binary form must reproduce the above copyright notice,
 this list of conditions and the following disclaimer in the documentation
 and/or other materials provided with the distribution.

 * The name of the copyright holders may not be used to endorse or promote products
 derived from this software without specific prior written permission.

 This software is provided by the copyright holders and contributors "as is" and
 any express or implied warranties, including, but not limited to, the implied
 warranties of merchantability and fitness for a particular purpose are disclaimed.
 In no event shall Gurpinder Singh Sandhu be liable for any direct,
 indirect, incidental, special, exemplary, or consequential damages
 (including, but not limited to, procurement of substitute goods or services;
 loss of use, data, or profits; or business interruption) however caused
 and on any theory of liability, whether in contract, strict liability,
 or tort (including negligence or otherwise) arising in any way out of
 the use of this software, even if advised of the possibility of such damage.*/
#include "endeffector_tracking/KernelBasedTracking.h"

void 
KernelBasedTracking::create_kernel(cv::Mat &kernel_)
{
	int ck_no_rows = kernel_.rows;
	int ck_no_cols = kernel_.cols;
	float ck_centre[2] = {float((ck_no_cols-1)/2), float((ck_no_rows-1)/2)};

	float parameter_cd = 0.1*pi1*ck_no_rows*ck_no_rows;

	for(int i=0;i<ck_no_rows;i++)
	{
		for(int j=0;j<ck_no_cols;j++)
		{
			float x = (abs(i-ck_centre[0]));
			float y = (abs(j-ck_centre[1]));

			float n = static_cast<float>(parameter_cd*(1.0-((x*x+y*y)/(ck_centre[0]*ck_centre[0]))));
			float m = n<0?0:n;
			kernel_.at<float>(i,j) = m;

		}
	}

	kernel_sum = 0;
	for(int i=0;i<ck_no_rows;i++)
	{
		for(int j=0;j<ck_no_cols;j++)
		{
			kernel_sum += kernel_.at<float>(i,j);
		}
	}
}

cv::Mat 
KernelBasedTracking::create_target_model(cv::Mat &input_image, cv::Mat &input_kernel)
{
	cv::Mat target_model(3,no_of_bins,CV_32F,cv::Scalar(1e-10));

	int ctm_no_cols = input_kernel.cols;
	int ctm_no_rows = input_kernel.rows;

	cv::Vec3f curr_pixel_value;
	cv::Vec3f bin_value;

	for(int i=0;i<ctm_no_rows;i++)
	{
		for(int j=0;j<ctm_no_cols;j++)
		{
			curr_pixel_value = input_image.at<cv::Vec3b>(i,j);
			bin_value[0] = curr_pixel_value[0]/bin_width;
			bin_value[1] = curr_pixel_value[1]/bin_width;
			bin_value[2] = curr_pixel_value[2]/bin_width;
			target_model.at<float>(0,bin_value[0]) += input_kernel.at<float>(i,j) / kernel_sum;
			target_model.at<float>(1,bin_value[1]) += input_kernel.at<float>(i,j) / kernel_sum;
			target_model.at<float>(2,bin_value[2]) += input_kernel.at<float>(i,j) / kernel_sum;
		}
	}

	return target_model;
}


cv::Mat 
KernelBasedTracking::detect_object(cv::Mat &input_image, cv::Rect &box_)
{
	// extract the box from the input_image
	cv::Mat extractedImg = cv::Mat(input_image, cv::Range(box_.y, box_.y + box_.height + 1), cv::Range(box_.x, box_.x + box_.width + 1));

	// resize the img to the size of the kernel
	cv::Mat resizedImg;
	cv::resize(extractedImg, resizedImg, kernel_for_now.size());

	// create the target model
	cv::Mat target_model = create_target_model(resizedImg, kernel_for_now);

	return target_model;
}

cv::Mat 
KernelBasedTracking::assign_weight(cv::Mat &input_image,cv::Mat &target_model,cv::Mat &target_candidate,cv::Rect &box_)
{
	int aw_no_of_rows = box_.height>box_.width?box_.width:box_.height;
	int aw_no_of_cols = box_.height>box_.width?box_.width:box_.height;

	cv::Mat weight_(aw_no_of_rows,aw_no_of_cols,CV_32F,cv::Scalar(1.0000));

	std::vector<cv::Mat> bgr_planes;

	split(input_image, bgr_planes);

	int i_img = box_.y;
	int j_img = box_.x;

	int curr_pixel = 0;
	int bin_value = 0;

	for(int k = 0; k < 3;  k++)
	{	
		i_img = box_.y;
		for(int i = 0;i<aw_no_of_rows;i++)
		{
			j_img = box_.x;
			for(int j = 0;j<aw_no_of_cols;j++)
			{
				curr_pixel = static_cast<int>(bgr_planes[k].at<uchar>(i_img,j_img));
				bin_value = int(curr_pixel / bin_width);

				weight_.at<float>(i,j) *= static_cast<float>((sqrt(target_model.at<float>(k,bin_value)/target_candidate.at<float>(k,bin_value))));

			j_img++;		
			}

		i_img++;
		}
	}

	return weight_;
}

float 
KernelBasedTracking::calc_bhattacharya(cv::Mat &target_model,cv::Mat &target_candidate)
{
	float p_bar = 0.0;
	float dist = 0.0;
	float target_sum = 0.0;
	float candidate_sum = 0.0;

	for(int i = 0; i < target_model.rows; i++)
	{
		for (int j = 0; j < target_model.cols; j++)
		{
			target_sum += target_model.at<float>(i,j);
			candidate_sum += target_candidate.at<float>(i,j);
		}
	}
	target_sum = sqrt(target_sum);
	candidate_sum = sqrt(candidate_sum);

	for(int i = 0; i < target_model.rows; i++)
		for (int j = 0; j < target_model.cols; j++)
			p_bar += sqrt(target_candidate.at<float>(i,j)*target_model.at<float>(i,j)) / target_sum / candidate_sum;

	dist = sqrt(1.0-p_bar);

	return dist;

}

void
KernelBasedTracking::track()
{
	frame_count++;

	cv::Rect box1, box2, box3;
	cv::Mat candidate1, candidate2, candidate3;
	float dist1, dist2, dist3;

	//dist1 = trackscale(0.99, box1, candidate1);
	dist1 = 2;
	dist2 = trackscale(1.0, box2, candidate2);
	dist3 = 2;
	//dist3 = trackscale(1.01, box3, candidate3);

	float dist;
	if (dist1 < dist2 && dist1 < dist3)
	{
		dist = dist1;
		box = box1;
		target_candidate_hist = candidate1;
	}
	else if (dist2 < dist1 && dist2 < dist3)
	{
		dist = dist2;
		box = box2;
		target_candidate_hist = candidate2;
	}
	else
	{
		dist = dist3;
		box = box3;
		target_candidate_hist = candidate3;
	}

	/*
	if(dist < 0.01 && frame_count > 30)
	{	
		float w = 0.1;
		target_hist = w * target_candidate_hist + (1 - w) * target_hist;
		frame_count = 0;
	}
	*/

}

KernelBasedTracking::KernelBasedTracking()
:kernel_sum(0)
,pi1(3.1415927)
,frame_count(0)
,model(0)
,alpha(0.9)
,no_of_bins(16)
,range(256)
{
	bin_width = cvRound(float(range)/float(no_of_bins)); 
}

void
KernelBasedTracking::retrieveImage(const cv::Mat& img)
{
	orig_img = img.clone();
}

void
KernelBasedTracking::initTarget(const int *win_)
{
	box.x 		= win_[1];
	box.y 		= win_[0];
	box.width 	= win_[3] - win_[1];
	box.height 	= win_[2] - win_[0];

	kernel_size = box.height>box.width?box.width:box.height;
	kernel_for_now = cv::Mat(kernel_size,kernel_size,CV_32F,cv::Scalar(0));
	create_kernel(kernel_for_now);

	if(box.width%2==0)
		box.width++;
	if(box.height%2==0)
		box.height++;

	target_hist = detect_object(orig_img,box);
}

cv::Rect
KernelBasedTracking::getWindow(void)
{
	return box;
}

float 
KernelBasedTracking::trackscale(float scale, cv::Rect& box_, cv::Mat& target_candidate)
{
	box_ = this->box;
	box_.height = int(box_.height * scale);
	box_.width  = int(box_.width * scale);

	cv::Rect next_box;
	for(int k=0;k<10;k++)
	{
		target_candidate = detect_object(orig_img,box_);
		weight = assign_weight(orig_img,target_hist,target_candidate,box_);

		float num_x = 0.0;
		float den = 0.0;
		float num_y = 0.0;
		float centre = static_cast<float>((weight.rows-1)/2.0);
		double mult = 0.0;
		float norm_i = 0.0;
		float norm_j = 0.0;

		next_box.x = box_.x;
		next_box.y = box_.y;
		next_box.width = box_.width;
		next_box.height = box_.height;

		for(int i=0;i<weight.rows;i++)
		{
			for(int j=0;j<weight.cols;j++)
			{
				norm_i = static_cast<float>(i-centre)/centre;
				norm_j = static_cast<float>(j-centre)/centre;
				mult = pow(norm_i,2)+pow(norm_j,2)>1.0?0.0:1.0;
				num_x += static_cast<float>(alpha*norm_j*weight.at<float>(i,j)*mult);
				num_y += static_cast<float>(alpha*norm_i*weight.at<float>(i,j)*mult);

				den += static_cast<float>(weight.at<float>(i,j)*mult);
			}
		}

		next_box.x += static_cast<int>((num_x/den)*centre);
		next_box.y += static_cast<int>((num_y/den)*centre);


		if(next_box.x-box_.x>5)
		{
			next_box.x = box_.x + 5;
		}
		if(next_box.x-box_.x<-5)
		{
			next_box.x = box_.x - 5;
		}
		if(next_box.y-box_.y>5)
		{
			next_box.y = box_.y + 5;
		}
		if(next_box.y-box_.y<-5)
		{
			next_box.y = box_.y - 5;
		}

		if(abs(next_box.x-box_.x)<2 && abs(next_box.y-box_.y)<2)
		{
			break;
		}
		else
		{
			box_.x = next_box.x;
			box_.y = next_box.y;
		}
		// TODO:add Tracking lost procedure here
		if(box_.x + box_.width >= orig_img.cols || box_.x <= 0 || box_.y + box_.height >= orig_img.rows || box_.y <= 0)
		{
			//do_something()
			int a;
			std::cin>>a;
		}

	}

	float dist = calc_bhattacharya(target_hist,target_candidate);

	return dist;
}
