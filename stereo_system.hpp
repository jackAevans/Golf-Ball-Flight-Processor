#pragma once

#include <opencv2/opencv.hpp>

struct StereoSystem {
    cv::Mat K1, D1, K2, D2;       // Intrinsics + distortion
    cv::Mat R, T;                 // Extrinsics
    cv::Mat R1, R2, P1, P2, Q;    // Rectification & projection matrices
    cv::Size imageSize;
};

StereoSystem createBlenderStereoSystem(
    cv::Point3f leftCameraPos, 
    float baseline,            
    float fov,
    cv::Size imageSize,
    cv::Point2f principalPoint = cv::Point2f(-1,-1)
);

StereoSystem calibrateStereoSystem(
    const std::vector<cv::Mat>& leftImgs, 
    const std::vector<cv::Mat>& rightImgs,
    const cv::Size& boardSize, 
    float squareSize
);

std::string serializeStereoSystem(const StereoSystem &stereoSystem);

StereoSystem deserializeStereoSystem(const std::string &data);

cv::Point3f triangulatePoint(cv::Point2d leftPoint, cv::Point2d rightPoint, const StereoSystem& stereoSystem);