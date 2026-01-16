#pragma once

#include <opencv2/opencv.hpp>

cv::Mat removeBackground(const cv::Mat &img, const cv::Mat &backgroundImg);

cv::Mat cropSquarePatch(const cv::Mat& src, cv::Point2f center, int cropSize);

cv::Mat normalizeTopPercentile(const cv::Mat& src, float topPercent);

std::vector<cv::Point2f> detectBlobs(
    cv::Mat img, 
    int minPixelWidth, 
    int maxPixelWidth, 
    bool findRound, 
    int colour = 255
);

cv::RotatedRect findBlobEllipse(const cv::Mat& src, int minArea, int maxArea);

struct PreprocessStages {
    cv::Mat gray;
    cv::Mat blurred;
    cv::Mat normalized;
    cv::Mat thresholded;
};

PreprocessStages generatePreprocessedStages(
    const cv::Mat& src, 
    int blurSize, 
    float normalizeTopPercent, 
    int thresholdBias
);

// ---DEBUG---

void drawPoints(
    cv::Mat& image, 
    const std::vector<cv::Point2f>& points, 
    cv::Scalar colour = cv::Scalar(0, 0, 255), 
    bool diffColour = false
);

void drawEllipseDetection(cv::Mat& image, cv::RotatedRect ellipse);

cv::Mat maskImageWithEllipse(const cv::Mat& src, const cv::RotatedRect& ellipse);

