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

cv::Point3f triangulatePoint(cv::Point2f leftPoint, cv::Point2f rightPoint, const StereoSystem& stereoSystem);

void getRay(
    const StereoSystem &stereoSystem, 
    const cv::Point2f &coordinate, 
    int cameraID, 
    cv::Point3f &origin, 
    cv::Point3f &direction
);

bool validateStereoMatch(
    cv::Point2f leftPoint, 
    cv::Point2f rightPoint, 
    float minZ, float maxZ, 
    const StereoSystem& stereoSystem,
    float maxYDiff = 3.0f 
);

void matchStereoPoints(
    std::vector<cv::Point2f>& leftPoints,
    std::vector<cv::Point2f>& rightPoints,
    float minZ, float maxZ,
    const StereoSystem& stereoSystem,
    float maxYDiff = 3.0f
);