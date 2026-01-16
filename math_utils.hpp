#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

std::vector<cv::Point3f> intersectRaySphere(
    const cv::Point3f &origin, 
    const cv::Point3f &direction, 
    const cv::Point3f &center,
    float radius
);

std::vector<cv::Mat> computeBestMatchRotations(
    const std::vector<cv::Point3f>& src,
    const std::vector<cv::Point3f>& dst
);

std::vector<cv::Point3f> rotatePoints(
    const std::vector<cv::Point3f> &points, 
    const cv::Mat &R
);