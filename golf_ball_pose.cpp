#include "golf_ball_pose.hpp"
#include "image_processing.hpp"

std::vector<GolfBallPose> estimateGolfBallPoses(
    const cv::Mat &leftImage, 
    const cv::Mat &rightImage,
    const StereoSystem &stereoSystem,
    const GolfBallPoseEstimationParameters &params,
    bool debug
){
    const float MAX_Y_DIFF = 20;
    const int BLUR_SIZE = 3;
    const float NORMALIZED_TOP_PERCENT = 0.1;
    const int THRESHOLD_BIAS = -20;
    const int MIN_ELLIPSE_AREA = M_PI * ((params.minBallSize_px/2) * (params.minBallSize_px/2));
    const int MAX_ELLIPSE_AREA = M_PI * ((params.maxBallSize_px/2) * (params.maxBallSize_px/2));

    cv::Mat leftDebug;
    cv::Mat rightDebug;

    if(debug){  
        leftDebug = leftImage.clone();
        rightDebug = rightImage.clone();
    }

    std::vector<cv::Point2f> leftBlobs = detectBlobs(leftImage, params.minBallSize_px, params.maxBallSize_px, true);
    std::vector<cv::Point2f> rightBlobs = detectBlobs(rightImage, params.minBallSize_px, params.maxBallSize_px, true);

    matchStereoPoints(leftBlobs, rightBlobs, params.minBallDepth, params.maxBallDepth, stereoSystem, MAX_Y_DIFF);

    for(std::size_t i = 0; i < leftBlobs.size(); i++){
        cv::Point2f leftBlob = leftBlobs[i];
        cv::Point2f rightBlob = rightBlobs[i];

        cv::Mat leftCropped = cropSquarePatch(leftImage, leftBlob, params.maxBallSize_px);
        cv::Mat rightCropped = cropSquarePatch(rightImage, rightBlob, params.maxBallSize_px);

        PreprocessStages ppsLeft = generatePreprocessedStages(leftCropped, BLUR_SIZE, NORMALIZED_TOP_PERCENT, THRESHOLD_BIAS);
        PreprocessStages ppsRight = generatePreprocessedStages(rightCropped, BLUR_SIZE, NORMALIZED_TOP_PERCENT, THRESHOLD_BIAS);


        cv::RotatedRect ellipesLeft = findBlobEllipse(ppsLeft.thresholded, MIN_ELLIPSE_AREA, MAX_ELLIPSE_AREA);
        cv::RotatedRect ellipesRight = findBlobEllipse(ppsRight.thresholded, MIN_ELLIPSE_AREA, MAX_ELLIPSE_AREA);

        ellipesLeft.center += leftBlob;
        ellipesRight.center += rightBlob;

        cv::Point3f golfBallPosition = triangulatePoint(ellipesLeft.center, ellipesRight.center, stereoSystem);

        std::vector<cv::Point2f> leftDotLocations = detectBlobs(
            ppsLeft.normalized, 
            params.minDotSize_px, 
            params.maxBallSize_px, 
            false, 0
        );

        for(cv::Point2f &dot: leftDotLocations){
            dot += leftBlob - cv::Point2f(params.maxBallSize_px/2, params.maxBallSize_px/2);
        }

        std::vector<cv::Point2f> rightDotLocations = detectBlobs(
            ppsRight.normalized, 
            params.minDotSize_px, 
            params.maxBallSize_px, 
            false, 0
        );

        for(cv::Point2f &dot: rightDotLocations){
            dot += rightBlob - cv::Point2f(params.maxBallSize_px/2, params.maxBallSize_px/2);
        }

        if(debug){
            drawPoints(leftDebug, leftDotLocations);
            drawPoints(rightDebug, rightDotLocations);

            drawEllipseDetection(leftDebug, ellipesLeft);
            drawEllipseDetection(rightDebug, ellipesRight);
        }

    }

    if(debug){
        cv::imshow("left image", leftDebug);
        cv::imshow("right image", rightDebug);
        cv::waitKey(0);
    }

    return {};
}