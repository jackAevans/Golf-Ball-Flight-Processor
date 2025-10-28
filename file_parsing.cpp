#include "file_parsing.hpp"

#include <filesystem>
#include <fstream>
#include <string>

namespace fs = std::filesystem;

std::vector<cv::Mat> loadImages(const std::string& directoryPath) {
    std::vector<cv::Mat> images;
    std::vector<std::string> paths;

    try {
        // First, collect all file paths
        for (const auto& entry : fs::directory_iterator(directoryPath)) {
            if (entry.is_regular_file()) {
                paths.push_back(entry.path().string());
            }
        }

        // Sort paths by filename (lexicographical order)
        std::sort(paths.begin(), paths.end());

        // Load in sorted order
        for (const auto& p : paths) {
            cv::Mat img = cv::imread(p, cv::IMREAD_COLOR);
            if (img.empty()) {
                std::cerr << "Warning: Could not load image: " << p << "\n";
                continue;
            }
            images.push_back(img);
        }
    } catch (const fs::filesystem_error& e) {
        std::cerr << "Filesystem error: " << e.what() << "\n";
    }

    return images;
}

void writeToFile(const std::string& filename, const std::string& content) {
    // Open the file with truncation (creates new or overwrites existing)
    std::ofstream outFile(filename);
    
    if (!outFile) {
        std::cerr << "Error: Could not open file " << filename << " for writing.\n";
        return;
    }

    outFile << content;  // Write the string to the file
    outFile.close();     // Close the file
}

std::string readFromFile(const std::string& filename) {
    std::ifstream inFile(filename);
    
    if (!inFile) {
        std::cerr << "Error: Could not open file " << filename << " for reading.\n";
        return "";
    }

    std::stringstream buffer;
    buffer << inFile.rdbuf();  // Read entire file into buffer
    return buffer.str();       // Return as std::string
}