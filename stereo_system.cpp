#include "stereo_system.hpp"

StereoSystem createBlenderStereoSystem(
    cv::Point3f leftCameraPos,  // position of the left camera in Blender world
    float baseline,            // distance between cameras in meters (along Blender X)
    float FOV,
    cv::Size imageSize,
    cv::Point2f principalPoint
){
    StereoSystem ss;
    ss.imageSize = imageSize;

    // --- Intrinsics ---
    double fx = (imageSize.width / 2.0) / std::tan((FOV * CV_PI / 180.0) / 2.0);
    fx *= 1.008;
    double fy = fx; // square pixels assumption
    if (principalPoint.x < 0) principalPoint.x = imageSize.width / 2.0;
    if (principalPoint.y < 0) principalPoint.y = imageSize.height / 2.0;

    ss.K1 = (cv::Mat_<double>(3,3) << fx, 0, principalPoint.x,
                                      0, fy, principalPoint.y,
                                      0, 0, 1);
    ss.K2 = ss.K1.clone();

    // --- Lens distortion ---
    ss.D1 = cv::Mat::zeros(1,5,CV_64F);
    ss.D2 = cv::Mat::zeros(1,5,CV_64F);

    // --- Compute right camera position in Blender ---
    cv::Point3f rightCameraPos = leftCameraPos + cv::Point3f(baseline, 0.0, 0.0);

    // --- Convert Blender coordinates to OpenCV coordinates ---
    auto blenderToCV = [](const cv::Point3f &p) {
        return cv::Point3f(p.x, -p.z, p.y);
    };

    cv::Point3f leftCV  = blenderToCV(leftCameraPos);
    cv::Point3f rightCV = blenderToCV(rightCameraPos);

    // --- Extrinsics ---
    cv::Point3f tvec = rightCV - leftCV;
    ss.T = (cv::Mat_<double>(3,1) << tvec.x, tvec.y, tvec.z);

    // Both cameras look along +Z in OpenCV → rotation is identity
    ss.R = cv::Mat::eye(3,3,CV_64F);

    // --- Rectification ---
    cv::stereoRectify(
        ss.K1, ss.D1,
        ss.K2, ss.D2,
        imageSize,
        ss.R, ss.T,
        ss.R1, ss.R2, ss.P1, ss.P2, ss.Q,
        cv::CALIB_ZERO_DISPARITY, 0, imageSize);

    return ss;
}

StereoSystem calibrateStereoSystem(
    const std::vector<cv::Mat>& leftImgs,
    const std::vector<cv::Mat>& rightImgs,
    const cv::Size& boardSize, 
    float squareSizeMeters
){
    StereoSystem stereoSystem;

    std::vector<std::vector<cv::Point3f>> objectPointsMono; 
    std::vector<std::vector<cv::Point2f>> imgPointsLeft, imgPointsRight;

    // Prepare the chessboard 3D points (Z=0 plane)
    std::vector<cv::Point3f> obj;
    for (int i = 0; i < boardSize.height; ++i) {
        for (int j = 0; j < boardSize.width; ++j) {
            obj.emplace_back(j * squareSizeMeters, i * squareSizeMeters, 0.0f);
        }
    }

    // Detect corners on all stereo pairs
    for (size_t i = 0; i < leftImgs.size(); ++i) {
        std::vector<cv::Point2f> cornersLeft, cornersRight;
        bool foundLeft  = cv::findChessboardCorners(leftImgs[i], boardSize, cornersLeft);
        bool foundRight = cv::findChessboardCorners(rightImgs[i], boardSize, cornersRight);

        if (foundLeft && foundRight) {
            cv::Mat grayLeft, grayRight;
            cv::cvtColor(leftImgs[i], grayLeft, cv::COLOR_BGR2GRAY);
            cv::cvtColor(rightImgs[i], grayRight, cv::COLOR_BGR2GRAY);

            cv::cornerSubPix(grayLeft, cornersLeft, cv::Size(11,11), cv::Size(-1,-1),
                             cv::TermCriteria(cv::TermCriteria::EPS+cv::TermCriteria::COUNT, 30, 0.01));
            cv::cornerSubPix(grayRight, cornersRight, cv::Size(11,11), cv::Size(-1,-1),
                             cv::TermCriteria(cv::TermCriteria::EPS+cv::TermCriteria::COUNT, 30, 0.01));

            imgPointsLeft.push_back(cornersLeft);
            imgPointsRight.push_back(cornersRight);
            objectPointsMono.push_back(obj);

            // Optional: visualize
            cv::Mat leftDisp = leftImgs[i].clone();
            cv::Mat rightDisp = rightImgs[i].clone();
            cv::drawChessboardCorners(leftDisp, boardSize, cornersLeft, foundLeft);
            cv::drawChessboardCorners(rightDisp, boardSize, cornersRight, foundRight);
            cv::imshow("Left", leftDisp);
            cv::imshow("Right", rightDisp);
            if (cv::waitKey(100) == 27) break;
        }
    }

    stereoSystem.imageSize = leftImgs[0].size();

    // Monocular calibration for left and right cameras
    std::vector<cv::Mat> rvecs, tvecs;
    double rmsLeft = cv::calibrateCamera(objectPointsMono, imgPointsLeft, stereoSystem.imageSize,
                                         stereoSystem.K1, stereoSystem.D1, rvecs, tvecs);
    double rmsRight = cv::calibrateCamera(objectPointsMono, imgPointsRight, stereoSystem.imageSize,
                                          stereoSystem.K2, stereoSystem.D2, rvecs, tvecs);

    std::cout << "Left camera RMS reprojection error = " << rmsLeft << "\n";
    std::cout << "Right camera RMS reprojection error = " << rmsRight << "\n";

    // Stereo calibration (extrinsics only)
    cv::Mat E, F;
    double rmsStereo = cv::stereoCalibrate(
        objectPointsMono, imgPointsLeft, imgPointsRight,
        stereoSystem.K1, stereoSystem.D1,
        stereoSystem.K2, stereoSystem.D2,
        stereoSystem.imageSize,
        stereoSystem.R, stereoSystem.T,
        E, F,
        cv::CALIB_FIX_INTRINSIC,
        cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 100, 1e-5)
    );

    std::cout << "Stereo calibration RMS = " << rmsStereo << "\n";

    // Stereo rectification: fill R1, R2, P1, P2, Q
    cv::stereoRectify(
        stereoSystem.K1, stereoSystem.D1,
        stereoSystem.K2, stereoSystem.D2,
        stereoSystem.imageSize,
        stereoSystem.R, stereoSystem.T,
        stereoSystem.R1, stereoSystem.R2,
        stereoSystem.P1, stereoSystem.P2,
        stereoSystem.Q,
        cv::CALIB_ZERO_DISPARITY, // or 0
        1,                        // alpha, 1 = all pixels retained
        stereoSystem.imageSize
    );

    return stereoSystem;
}

std::string serializeStereoSystem(const StereoSystem &stereoSystem) {
    cv::FileStorage fs("", cv::FileStorage::WRITE | cv::FileStorage::MEMORY);

    fs << "K1" << stereoSystem.K1;
    fs << "D1" << stereoSystem.D1;
    fs << "K2" << stereoSystem.K2;
    fs << "D2" << stereoSystem.D2;
    fs << "R"  << stereoSystem.R;
    fs << "T"  << stereoSystem.T;
    fs << "R1" << stereoSystem.R1;
    fs << "R2" << stereoSystem.R2;
    fs << "P1" << stereoSystem.P1;
    fs << "P2" << stereoSystem.P2;
    fs << "Q"  << stereoSystem.Q;
    fs << "imageSize_width"  << stereoSystem.imageSize.width;
    fs << "imageSize_height" << stereoSystem.imageSize.height;

    return fs.releaseAndGetString();
}

StereoSystem deserializeStereoSystem(const std::string &data){
    StereoSystem ss;
    cv::FileStorage fs(data, cv::FileStorage::READ | cv::FileStorage::MEMORY);

    fs["K1"] >> ss.K1;
    fs["D1"] >> ss.D1;
    fs["K2"] >> ss.K2;
    fs["D2"] >> ss.D2;
    fs["R"]  >> ss.R;
    fs["T"]  >> ss.T;
    fs["R1"] >> ss.R1;
    fs["R2"] >> ss.R2;
    fs["P1"] >> ss.P1;
    fs["P2"] >> ss.P2;
    fs["Q"]  >> ss.Q;
    fs["imageSize_width"]  >> ss.imageSize.width;
    fs["imageSize_height"] >> ss.imageSize.height;

    return ss;
}

cv::Point3f triangulatePoint(cv::Point2f leftPoint, cv::Point2f rightPoint, const StereoSystem& stereoSystem){
    // Wrap pixel points into vector<cv::Point2f>
    std::vector<cv::Point2f> ptsLeft = {leftPoint};
    std::vector<cv::Point2f> ptsRight = {rightPoint};

    // Undistort and rectify points:
    // Pass the rectification and projection matrices to get normalized rectified points ready for triangulation
    std::vector<cv::Point2f> rectLeft, rectRight;
    cv::undistortPoints(ptsLeft, rectLeft, stereoSystem.K1, stereoSystem.D1, stereoSystem.R1, stereoSystem.P1);
    cv::undistortPoints(ptsRight, rectRight, stereoSystem.K2, stereoSystem.D2, stereoSystem.R2, stereoSystem.P2);

    // Triangulate points using rectified projection matrices (3x4)
    cv::Mat point4D;
    cv::triangulatePoints(stereoSystem.P1, stereoSystem.P2, rectLeft, rectRight, point4D);

    // Convert homogeneous coordinates to 3D
    float w = point4D.at<float>(3, 0);
    cv::Point3f point3f(
        point4D.at<float>(0, 0) / w,
        point4D.at<float>(1, 0) / w,
        point4D.at<float>(2, 0) / w
    );

    return point3f;  // No sign flip needed unless your coordinate system requires it
}

void getRay(
    const StereoSystem &stereoSystem, 
    const cv::Point2f &coordinate, 
    int cameraID, 
    cv::Point3f &origin, 
    cv::Point3f &direction
){
    // 1. Choose camera parameters
    cv::Mat K = (cameraID == 1) ? stereoSystem.K2 : stereoSystem.K1;
    cv::Mat D = (cameraID == 1) ? stereoSystem.D2 : stereoSystem.D1;
    cv::Mat R_cam = cv::Mat::eye(3, 3, CV_64F);
    cv::Mat t_cam = cv::Mat::zeros(3, 1, CV_64F);

    if (cameraID == 1) {
        // For right camera, extrinsics transform from right camera to left camera/world
        R_cam = stereoSystem.R.clone();
        t_cam = stereoSystem.T.clone();
    }

    // 2. Undistort pixel to normalized image coordinates
    std::vector<cv::Point2f> srcPt(1, coordinate), undistPt;
    cv::undistortPoints(srcPt, undistPt, K, D);

    // undistPt is in normalized camera coordinates (z=1)
    cv::Point3f dirCam(undistPt[0].x, undistPt[0].y, 1.0f);

    // 3. Transform ray to world/left camera space
    cv::Mat dirCamMat = (cv::Mat_<double>(3, 1) << dirCam.x, dirCam.y, dirCam.z);
    cv::Mat dirWorldMat = R_cam.t() * dirCamMat; // rotate to world
    cv::Mat originWorldMat = -R_cam.t() * t_cam; // camera center in world

    // 4. Normalize direction
    cv::Point3f dirWorld(
        (float)dirWorldMat.at<double>(0),
        (float)dirWorldMat.at<double>(1),
        (float)dirWorldMat.at<double>(2)
    );
    float len = std::sqrt(dirWorld.x*dirWorld.x + dirWorld.y*dirWorld.y + dirWorld.z*dirWorld.z);
    dirWorld.x /= len;
    dirWorld.y /= len;
    dirWorld.z /= len;

    cv::Point3f originWorld(
        (float)originWorldMat.at<double>(0),
        (float)originWorldMat.at<double>(1),
        (float)originWorldMat.at<double>(2)
    );

    origin = originWorld;
    direction = dirWorld;
}

bool validateStereoMatch(
    cv::Point2f leftPoint, 
    cv::Point2f rightPoint, 
    float minZ, float maxZ, 
    const StereoSystem& stereoSystem,
    float maxYDiff
) {
    // Wrap pixel points into vector<cv::Point2f>
    std::vector<cv::Point2f> ptsLeft = { leftPoint };
    std::vector<cv::Point2f> ptsRight = { rightPoint };

    // Undistort and rectify points
    std::vector<cv::Point2f> rectLeft, rectRight;
    cv::undistortPoints(ptsLeft, rectLeft, stereoSystem.K1, stereoSystem.D1, stereoSystem.R1, stereoSystem.P1);
    cv::undistortPoints(ptsRight, rectRight, stereoSystem.K2, stereoSystem.D2, stereoSystem.R2, stereoSystem.P2);

    // Check y-values for epipolar consistency
    if (std::abs(rectLeft[0].y - rectRight[0].y) > maxYDiff) {
        return false; // Not epipolar-consistent
    }

    // Triangulate the 3D point
    cv::Mat point4D;
    cv::triangulatePoints(stereoSystem.P1, stereoSystem.P2, rectLeft, rectRight, point4D);

    // Convert from homogeneous coordinates to 3D
    cv::Vec3f point3f(
        point4D.at<float>(0) / point4D.at<float>(3),
        point4D.at<float>(1) / point4D.at<float>(3),
        point4D.at<float>(2) / point4D.at<float>(3)
    );

    // Check depth (Z) range
    if (point3f[2] < minZ || point3f[2] > maxZ) {
        return false;
    }

    // Passed all checks
    return true;
}

void matchStereoPoints(
    std::vector<cv::Point2f>& leftPoints,
    std::vector<cv::Point2f>& rightPoints,
    float minZ, float maxZ,
    const StereoSystem& stereoSystem,
    float maxYDiff
) {
    // Vectors to hold matched points
    std::vector<cv::Point2f> matchedLeft, matchedRight;

    // Keep track of which right points are already matched
    std::vector<bool> rightMatched(rightPoints.size(), false);

    for (const auto& lp : leftPoints) {
        float bestDist = std::numeric_limits<float>::max();
        int bestIdx = -1;

        // Brute-force check against all unmatched right points
        for (size_t j = 0; j < rightPoints.size(); ++j) {
            if (rightMatched[j]) continue;

            if (validateStereoMatch(lp, rightPoints[j], minZ, maxZ, stereoSystem, maxYDiff)) {
                // Use simple horizontal distance as tie-breaker (optional)
                float dx = lp.x - rightPoints[j].x;
                float dy = lp.y - rightPoints[j].y;
                float dist = dx*dx + dy*dy;

                if (dist < bestDist) {
                    bestDist = dist;
                    bestIdx = (int)j;
                }
            }
        }

        // If a match was found, store it
        if (bestIdx != -1) {
            matchedLeft.push_back(lp);
            matchedRight.push_back(rightPoints[bestIdx]);
            rightMatched[bestIdx] = true; // mark as used
        }
    }

    // Replace input vectors with matched pairs
    leftPoints = matchedLeft;
    rightPoints = matchedRight;
}
