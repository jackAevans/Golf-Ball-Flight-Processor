#include "image_processing.hpp"

cv::Mat removeBackground(const cv::Mat &img, const cv::Mat &backgroundImg){
    // Ensure the images are the same size and type
    if (img.size() != backgroundImg.size() || img.type() != backgroundImg.type()) {
        throw std::runtime_error("Image and background must have the same size and type.");
    }

    cv::Mat diff;
    cv::absdiff(img, backgroundImg, diff);
    return diff;
}

cv::Mat cropSquarePatch(const cv::Mat& src, cv::Point2f center, int cropSize){
    CV_Assert(!src.empty());
    CV_Assert(cropSize > 0);

    int half = cropSize / 2;

    // Round to nearest pixel center
    int cx = cvRound(center.x);
    int cy = cvRound(center.y);

    // Top-left of desired crop in source coordinates
    int x0 = cx - half;
    int y0 = cy - half;

    // Create output patch initialized to borderColor (black by default)
    cv::Mat patch(cropSize, cropSize, src.type(), cv::Scalar::all(0));

    // Compute overlap between desired crop rect and source image
    int srcX0 = std::max(0, x0);
    int srcY0 = std::max(0, y0);
    int srcX1 = std::min(src.cols, x0 + cropSize);
    int srcY1 = std::min(src.rows, y0 + cropSize);

    int overlapW = srcX1 - srcX0;
    int overlapH = srcY1 - srcY0;

    if (overlapW > 0 && overlapH > 0) {
        // Destination location inside patch where we paste the overlap
        int dstX0 = srcX0 - x0;
        int dstY0 = srcY0 - y0;

        cv::Rect srcR(srcX0, srcY0, overlapW, overlapH);
        cv::Rect dstR(dstX0, dstY0, overlapW, overlapH);

        src(srcR).copyTo(patch(dstR));
    }

    return patch;
}

cv::Mat normalizeTopPercentile(const cv::Mat& src, float topPercent) {
    using namespace cv;
    using namespace std;

    // Convert to grayscale if needed
    Mat gray;
    if (src.channels() == 3) {
        cvtColor(src, gray, COLOR_BGR2GRAY);
    } else {
        gray = src.clone();
    }

    // Flatten image into a vector
    vector<uchar> pixels;
    pixels.assign(gray.datastart, gray.dataend);

    // Sort pixel values
    sort(pixels.begin(), pixels.end());

    // Find the value at the 90th percentile
    size_t idx = static_cast<size_t>((1.0 - topPercent) * pixels.size());
    uchar percentileVal = pixels[idx];

    // Scale image so that percentileVal maps to 255
    Mat normalized;
    gray.convertTo(normalized, CV_32F);  // use float for scaling
    normalized = normalized * (255.0 / percentileVal);

    // Clip values to 255
    normalized.setTo(255, normalized > 255);
    normalized.convertTo(normalized, CV_8U);

    return normalized;
}

std::vector<cv::Point2f> detectBlobs(
    cv::Mat img, 
    int minPixelWidth, 
    int maxPixelWidth, 
    bool findRound, 
    int colour
){
    // Set up SimpleBlobDetector parameters
    cv::SimpleBlobDetector::Params params;
    params.filterByColor = true;
    params.blobColor = colour;  // detect light blobs

    params.filterByArea = true;
    double minDiameter = minPixelWidth; // slightly smaller than ball
    double maxDiameter = maxPixelWidth; // slightly larger than ball
    params.minArea = CV_PI * (minDiameter/2) * (minDiameter/2);
    params.maxArea = CV_PI * (maxDiameter/2) * (maxDiameter/2);

    params.filterByCircularity = false;

    params.filterByInertia = findRound;
    params.minInertiaRatio = 0.4f;  // round shapes

    params.filterByConvexity = false;
    params.minConvexity = 0.2f;

    cv::Ptr<cv::SimpleBlobDetector> detector = cv::SimpleBlobDetector::create(params);

    // Detect blobs
    std::vector<cv::KeyPoint> keypoints;
    detector->detect(img, keypoints);

    std::vector<cv::Point2f> centers;

    for (size_t i = 0; i < keypoints.size(); ++i) {
        centers.push_back(cv::Point2f(keypoints[i].pt.x,keypoints[i].pt.y));
    }

    std::sort(centers.begin(), centers.end(), [](const cv::Point2d& a, const cv::Point2d& b) {
        return a.x < b.x;
    });

    return centers;
}

cv::RotatedRect findBlobEllipse(const cv::Mat& src, int minArea, int maxArea) {
    using namespace cv;
    // --- Find contours ---
    std::vector<std::vector<Point>> contours;
    findContours(src, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    if (contours.empty())
        return cv::RotatedRect(); // No blob found

    // --- Fit ellipse and validate ---
    cv::RotatedRect finalEllipse = cv::RotatedRect();
    finalEllipse.center += cv::Point2f(src.cols/2, src.rows/2);

    for (const auto& contour : contours) {
        if (contour.size() < 5) continue;
        RotatedRect ellipseBox = fitEllipse(contour);

        // Roughly circular filter
        float aspectRatio = ellipseBox.size.width / ellipseBox.size.height;
        if (aspectRatio < 0.8 || aspectRatio > 1.25) continue;

        double area = contourArea(contour);
        if (area < minArea || area > maxArea) continue;

        finalEllipse = ellipseBox;
        break;
    }

    finalEllipse.center -= cv::Point2f(src.cols/2, src.rows/2);

    return finalEllipse;
}

PreprocessStages generatePreprocessedStages(
    const cv::Mat& src, 
    int blurSize, 
    float normalizeTopPercent, 
    int thresholdBias
){
    using namespace cv;

    // --- Step 1: Convert to grayscale ---
    Mat gray;
    if (src.channels() == 3)
        cvtColor(src, gray, COLOR_BGR2GRAY);
    else if (src.channels() == 4)
        cvtColor(src, gray, COLOR_BGRA2GRAY);
    else
        gray = src.clone();

    // --- Step 2: Smooth image to reduce noise ---
    Mat blurImg;
    medianBlur(gray, blurImg, blurSize);

    // --- Step 3: Normalize intensity (custom function) ---
    Mat normImg = normalizeTopPercentile(blurImg, normalizeTopPercent);

    // --- Step 4: Threshold (Otsu + custom bias) ---
    Mat binary;
    double otsuThresh = threshold(normImg, binary, 0, 255, THRESH_BINARY | THRESH_OTSU);
    double customCutoff = otsuThresh + thresholdBias;
    threshold(normImg, binary, customCutoff, 255, THRESH_BINARY);

    return{gray, blurImg, normImg, binary};
}

//---DEBUG---

cv::Scalar indexToColor(int idx) {
    // deterministic: pick hue based on index
    int hue = (idx * 47) % 180; // cycle through OpenCV HSV hue range [0,179]
    cv::Mat hsv(1, 1, CV_8UC3, cv::Scalar(hue, 200, 255)); 
    cv::Mat bgr;
    cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);
    cv::Vec3b color = bgr.at<cv::Vec3b>(0, 0);
    return cv::Scalar(color[0], color[1], color[2]);
}

void drawPoints(
    cv::Mat& image, 
    const std::vector<cv::Point2f>& points, 
    cv::Scalar colour, 
    bool diffColour) 
{
    for (int i = 0; i < (int)points.size(); i++) {
        cv::Point center(cvRound(points[i].x), cvRound(points[i].y));
        cv::Scalar c = diffColour ? indexToColor(i) : colour;

        cv::circle(image, center, 3, c, -1, cv::LINE_AA);
    }
}

void drawEllipseDetection(cv::Mat& image, cv::RotatedRect ellipse){
    cv::ellipse(image, ellipse, cv::Scalar(0, 120, 255), 2);
    cv::circle(image, ellipse.center, 3, cv::Scalar(0, 0, 255), -1);
    cv::drawMarker(image, ellipse.center, cv::Scalar(0, 255, 0), cv::MARKER_CROSS, 10, 1, cv::LINE_AA);
}

cv::Mat maskImageWithEllipse(const cv::Mat& src, const cv::RotatedRect& ellipse) {
    // Create a black mask same size as source image
    cv::Mat mask = cv::Mat::zeros(src.size(), CV_8UC1);

    cv::RotatedRect ellipseTrans = ellipse;

    ellipseTrans.center += cv::Point2f(src.cols/2, src.rows/2);

    // Draw a filled white ellipse on the mask
    cv::ellipse(mask, ellipseTrans, cv::Scalar(255), cv::FILLED);

    // Create a white background
    cv::Mat whiteBg(src.size(), src.type(), cv::Scalar(255, 255, 255));

    // Copy the original image onto the white background using the mask
    cv::Mat result;
    src.copyTo(whiteBg, mask); // only copy where mask == 255
    result = whiteBg;

    return result;
}


