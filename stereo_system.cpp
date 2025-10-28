#include "stereo_system.hpp"

StereoSystem createBlenderStereoSystem(
    cv::Point3f leftCameraPos,  // position of the left camera in Blender world
    double baseline,            // distance between cameras in meters (along Blender X)
    double FOV,
    cv::Size imageSize,
    cv::Point2d principalPoint
){
    StereoSystem ss;
    ss.imageSize = imageSize;

    // --- Intrinsics ---
    double fx = (imageSize.width / 2.0) / std::tan((FOV * CV_PI / 180.0) / 2.0);
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