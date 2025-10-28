#pragma once

#include <opencv2/opencv.hpp>

std::vector<cv::Mat> loadImages(const std::string& directoryPath);

void writeToFile(const std::string& filename, const std::string& content);

std::string readFromFile(const std::string& filename);