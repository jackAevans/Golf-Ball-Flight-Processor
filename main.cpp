#include <iostream>

#include "stereo_system.hpp"
#include "file_parsing.hpp"
#include "image_processing.hpp"
#include "golf_ball_pose.hpp"

int main() {
    // std::vector<cv::Mat> images = loadImages("../assets/images/5-iron");
    // StereoSystem ss = deserializeStereoSystem(readFromFile("../assets/launchMonitorData.xml"));

    // // std::vector<cv::Mat> images = loadImages("../assets/images/renders");
    // // StereoSystem ss = createBlenderStereoSystem(cv::Point3f(0,0,0), -0.1, 70.0f, cv::Size(1280, 800));

    // for(std::size_t i = 0; i < images.size()/4; i++){
    //     cv::Mat leftImage = removeBackground(images.at(4*i), images.at(4*i + 2));
    //     cv::Mat rightImage = removeBackground(images.at(4*i + 1), images.at(4*i + 3));

    //     // cv::Mat leftImage = images.at(2*i);
    //     // cv::Mat rightImage = images.at(2*i + 1);

    //     GolfBallPoseEstimationParameters params{
    //         .ballDotPattern = {},
    //         .ballRadius = GOLFBALL_RADIUS_M,
    //         .minBallSize_px = 50,
    //         .maxBallSize_px = 150,
    //         .minDotSize_px = 10,
    //         .maxDotSize_px = 20,
    //         .maxBallDepth = 0.5,
    //         .minBallDepth = 0.2,
    //         .maxDotMatchingTolerance = 0.002
    //     };

    //     estimateGolfBallPoses(leftImage, rightImage, ss, params, true);
    // }

    std::vector<cv::Mat> images = loadImages("../assets/images/newImages");

    return 0;
}