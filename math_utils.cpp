#include "math_utils.hpp"


std::vector<cv::Point3f> intersectRaySphere(
    const cv::Point3f &origin, 
    const cv::Point3f &direction, 
    const cv::Point3f &center,
    float radius
){
    std::vector<cv::Point3f> intersections;

    // Vector from ray origin to sphere center
    cv::Point3f oc = origin - center;

    // Quadratic coefficients: a t^2 + b t + c = 0
    float a = direction.dot(direction); // should be 1 if normalized
    float b = 2.0f * oc.dot(direction);
    float c = oc.dot(oc) - radius * radius;

    float discriminant = b*b - 4*a*c;

    if (discriminant < 0.0f) {
        return intersections; // no intersection
    }

    float sqrtDisc = std::sqrt(discriminant);

    float t1 = (-b - sqrtDisc) / (2.0f * a);
    float t2 = (-b + sqrtDisc) / (2.0f * a);

    // Only add intersections with t >= 0 (in front of ray origin)
    if (t1 >= 0.0f) {
        intersections.push_back(origin + t1 * direction);
    }
    if (t2 >= 0.0f && t2 != t1) {
        intersections.push_back(origin + t2 * direction);
    }

    return intersections;
}

cv::Mat alignPointPair(
    const cv::Point3f &p1_src, 
    const cv::Point3f &p1_dst,
    const cv::Point3f &p2_src, 
    const cv::Point3f &p2_dst
){
    auto normalize = [](const cv::Point3f &v) {
        double n = cv::norm(v);
        return n > 1e-12 ? v / n : cv::Point3f(0,0,0);
    };

    cv::Point3f x_src = normalize(p2_src - p1_src);
    cv::Point3f z_src = normalize(p1_src.cross(p2_src));
    cv::Point3f y_src = z_src.cross(x_src);

    cv::Mat R_src = (cv::Mat_<double>(3,3) <<
                     x_src.x, y_src.x, z_src.x,
                     x_src.y, y_src.y, z_src.y,
                     x_src.z, y_src.z, z_src.z);

    cv::Point3f x_dst = normalize(p2_dst - p1_dst);
    cv::Point3f z_dst = normalize(p1_dst.cross(p2_dst));
    cv::Point3f y_dst = z_dst.cross(x_dst);

    cv::Mat R_dst = (cv::Mat_<double>(3,3) <<
                     x_dst.x, y_dst.x, z_dst.x,
                     x_dst.y, y_dst.y, z_dst.y,
                     x_dst.z, y_dst.z, z_dst.z);

    return R_dst * R_src.t(); // rotation from src -> dst
}

cv::Mat kabsch(
    const std::vector<cv::Point3f>& src,
    const std::vector<cv::Point3f>& dst
){
    CV_Assert(src.size() == dst.size() && src.size() >= 3);

    // --- Compute centroids ---
    cv::Point3f meanSrc(0,0,0), meanDst(0,0,0);
    for (size_t i = 0; i < src.size(); ++i) {
        meanSrc += src[i];
        meanDst += dst[i];
    }
    meanSrc *= 1.0f / src.size();
    meanDst *= 1.0f / dst.size();

    // --- Subtract centroids (center the points) ---
    cv::Mat A(src.size(), 3, CV_64F);
    cv::Mat B(src.size(), 3, CV_64F);
    for (size_t i = 0; i < src.size(); ++i) {
        A.at<double>(i,0) = src[i].x - meanSrc.x;
        A.at<double>(i,1) = src[i].y - meanSrc.y;
        A.at<double>(i,2) = src[i].z - meanSrc.z;

        B.at<double>(i,0) = dst[i].x - meanDst.x;
        B.at<double>(i,1) = dst[i].y - meanDst.y;
        B.at<double>(i,2) = dst[i].z - meanDst.z;
    }

    // --- Covariance matrix ---
    cv::Mat H = A.t() * B;

    // --- Singular Value Decomposition ---
    cv::Mat U, S, Vt;
    cv::SVD::compute(H, S, U, Vt);

    // --- Compute optimal rotation ---
    cv::Mat R = Vt.t() * U.t();

    // Ensure a proper right-handed coordinate system (det(R) = +1)
    if (cv::determinant(R) < 0) {
        Vt.row(2) *= -1;
        R = Vt.t() * U.t();
    }

    return R; // 3x3 rotation matrix
}

struct Match {
    int idxA;
    int idxB;
    float distance;
};

// Finds closest 1-to-1 matches between two sets of 3D points
std::vector<Match> matchClosestPoints(
    const std::vector<cv::Point3f>& setA,
    const std::vector<cv::Point3f>& setB,
    float maxDistance  // optional cutoff
) {
    std::vector<Match> allPairs;
    allPairs.reserve(setA.size() * setB.size());

    // --- Compute all pairwise distances ---
    for (int i = 0; i < (int)setA.size(); ++i) {
        for (int j = 0; j < (int)setB.size(); ++j) {
            float dx = setA[i].x - setB[j].x;
            float dy = setA[i].y - setB[j].y;
            float dz = setA[i].z - setB[j].z;
            float dist = std::sqrt(dx*dx + dy*dy + dz*dz);
            if (dist <= maxDistance)
                allPairs.push_back({i, j, dist});
        }
    }

    // --- Sort by distance ---
    std::sort(allPairs.begin(), allPairs.end(),
              [](const Match& a, const Match& b) { return a.distance < b.distance; });

    // --- Track assigned points ---
    std::vector<bool> usedA(setA.size(), false);
    std::vector<bool> usedB(setB.size(), false);
    std::vector<Match> matches;

    // --- Greedy selection ---
    for (const auto& m : allPairs) {
        if (!usedA[m.idxA] && !usedB[m.idxB]) {
            matches.push_back(m);
            usedA[m.idxA] = true;
            usedB[m.idxB] = true;
        }
    }

    return matches;
}

void countMatchingPoints(
    const std::vector<cv::Point3f> &points1, 
    const std::vector<cv::Point3f> &points2, 
    double tolerance, 
    int &count, 
    double &score
){
    count = 0;
    score = 100000;

    for (const auto &p1 : points1) {
        for (const auto &p2 : points2) {
            double dx = p1.x - p2.x;
            double dy = p1.y - p2.y;
            double dz = p1.z - p2.z;
            double dist = std::sqrt(dx*dx + dy*dy + dz*dz);

            if (dist <= tolerance) {
                count++;
                score += dist * dist;
                break; // prevent double-counting the same p1
            }
        }
    }

    score /= (double)count;
}

double distance(const cv::Point3f& p1, const cv::Point3f& p2) {
    double dx = p1.x - p2.x;
    double dy = p1.y - p2.y;
    double dz = p1.z - p2.z;
    return std::sqrt(dx*dx + dy*dy + dz*dz);
}

std::vector<cv::Mat> computeBestMatchRotations(
    const std::vector<cv::Point3f>& src,
    const std::vector<cv::Point3f>& dst,
    double tolerance
){
    struct RotationCandidate {
        cv::Mat R;
        double score;
    };

    std::vector<RotationCandidate> candidates;
    int bestCount = 0;

    for (std::size_t i = 0; i < src.size(); i++) {
        for (std::size_t j = 0; j < src.size(); j++) {
            if (i == j) continue;

            cv::Point3d p11 = src[i];
            cv::Point3d p21 = src[j];

            for (std::size_t k = 0; k < dst.size(); k++) {
                for (std::size_t l = 0; l < dst.size(); l++) {
                    if (k == l) continue;

                    cv::Point3d p12 = dst[k];
                    cv::Point3d p22 = dst[l];

                    double srcDistance = distance(p11, p21);
                    double dstDistance = distance(p12, p22);

                    if (std::abs(srcDistance - dstDistance) > (tolerance * 2)) {
                        continue;
                    }

                    cv::Mat R_ = alignPointPair(p11, p12, p21, p22);

                    std::vector<cv::Point3f> rotatedPoints_ = rotatePoints(src, R_);

                    std::vector<Match> matches = matchClosestPoints(rotatedPoints_, dst, tolerance);

                    std::vector<cv::Point3f> matchingSrc;
                    std::vector<cv::Point3f> matchingDst;

                    for(Match match: matches){
                        matchingSrc.push_back(src[match.idxA]);
                        matchingDst.push_back(dst[match.idxB]);
                    }

                    cv::Mat R = kabsch(matchingSrc, matchingDst);

                    std::vector<cv::Point3f> rotatedPoints = rotatePoints(src, R);

                    int count = 0;
                    double score = 100000;

                    countMatchingPoints(rotatedPoints, dst, tolerance, count, score);

                    if (count > bestCount) {
                        // Found a new best → reset list
                        bestCount = count;
                        candidates.clear();
                        candidates.push_back({R, score});
                    } else if (count == bestCount && count > 0) {
                        // Same match count → add candidate
                        candidates.push_back({R, score});
                    }
                }
            }
        }
    }

    if (candidates.empty()) {
        // fallback: identity rotation
        candidates.push_back({cv::Mat::eye(3, 3, CV_64F), 100000});
    }

    // Sort by score, then by smaller angle if scores are equal
    std::sort(candidates.begin(), candidates.end(),
              [](const RotationCandidate &a, const RotationCandidate &b) {
                  if (std::abs(a.score - b.score) > 1e-6) {
                      return a.score < b.score; // lower score wins
                  }
                  cv::Mat rvecA, rvecB;
                  cv::Rodrigues(a.R, rvecA);
                  cv::Rodrigues(b.R, rvecB);
                  return cv::norm(rvecA) < cv::norm(rvecB);
              });

    // Extract only matrices
    std::vector<cv::Mat> rotations;
    rotations.reserve(candidates.size());
    for (auto &c : candidates) {
        rotations.push_back(c.R);
    }

    return rotations;
}

std::vector<cv::Point3f> rotatePoints(
    const std::vector<cv::Point3f> &points, 
    const cv::Mat &R
){
    std::vector<cv::Point3f> rotated;
    rotated.reserve(points.size());

    for (const auto &p : points) {
        // Convert point to 3x1 matrix
        cv::Mat v = (cv::Mat_<double>(3,1) << p.x, p.y, p.z);
        // Apply rotation
        cv::Mat vr = R * v;
        // Store as Point3d
        rotated.emplace_back(vr.at<double>(0), vr.at<double>(1), vr.at<double>(2));
    }

    return rotated;
}