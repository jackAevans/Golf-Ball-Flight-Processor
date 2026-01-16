#pragma once

#include "stereo_system.hpp"

#define GOLFBALL_RADIUS_M 0.02135

struct GolfBallPose {
    cv::Point3f position;
    cv::Mat orientation;
    std::vector<cv::Point3f> dots;
    float confidence = 0;
};

struct GolfBallPoseEstimationParameters {
    const std::vector<cv::Point3f> &ballDotPattern;
    float ballRadius;
    int minBallSize_px;
    int maxBallSize_px;
    int minDotSize_px;
    int maxDotSize_px;
    float minBallDepth;
    float maxBallDepth;
    float maxDotMatchingTolerance;
};

std::vector<GolfBallPose> estimateGolfBallPoses(
    const cv::Mat &leftImage, 
    const cv::Mat &rightImage,
    const StereoSystem &stereoSystem,
    const GolfBallPoseEstimationParameters &golfBallPoseEstimationParameters,
    bool debug
);