
#import <opencv2/opencv.hpp>
#import <opencv2/imgcodecs/ios.h>
#import <opencv2/videoio/cap_ios.h>
#import "opencvCamera-Bridging-Header.h"
#import <math.h>

using namespace std;
using namespace cv;

@interface OpenCVWrapper() <CvVideoCameraDelegate> {
    CvVideoCamera* cvCamera;
}

@end

@implementation OpenCVWrapper

- (UIImage*)binarization:(UIImage*)image {
    Mat mat;
    UIImageToMat(image, mat);
    
    cvtColor(mat, mat, COLOR_BGR2GRAY);
    
    threshold(mat, mat, 116, 255, THRESH_BINARY);
    
    return MatToUIImage(mat);
}

- (UIImage*)binarization_otsu:(UIImage *)image {
    Mat mat;
    UIImageToMat(image, mat);
    
    cvtColor(mat, mat, COLOR_BGR2GRAY);
    
    double ret = threshold(mat, mat, 0, 255, THRESH_OTSU);
    printf("ret=%f", ret);
    
    return MatToUIImage(mat);
}

- (UIImage*)canny:(UIImage *)image {
    Mat mat;
    Mat out;
    UIImageToMat(image, mat);
    
    Canny(mat, out, 255, 100);
    
    return MatToUIImage(out);
}

- (UIImage*)resize:(UIImage *)image width:(int)width height:(int)height {
    Mat mat;
    UIImageToMat(image, mat);
    
    resize(mat, mat, cvSize(width, height));
    
    return MatToUIImage(mat);
}

- (UIImage *)imageFilter:(UIImage *)image {
    cv::Mat mat;
    
    UIImageToMat(image, mat);
    
    // 2値化
    cv::Mat gray;
    cv::Mat bin;
    
    cv::cvtColor(mat, gray, CV_BGR2GRAY);
    cv::threshold(gray, bin, 200, 255, CV_THRESH_BINARY_INV);
    // ビット反転
    cv::bitwise_not(bin, bin);
    cv::threshold(bin, bin, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
    
    // 2値化した画像から輪郭抽出
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    // RETR_EXTERNAL(最も外側の輪郭のみ抽出)
    // CHAIN_APPROX_TC89_L1(近似アルゴリズム)
    //cv::findContours(bin, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_TC89_L1);
    cv::findContours(bin, contours, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);
    
    Mat bin_image_copy = bin.clone();
    Mat element = Mat::ones(9, 9, CV_8UC1);
    double contour_len = 0;
    for (int i = 0; i < contours.size(); i++) {
        contour_len += cv::arcLength(contours.at(i), 0);
    }
    NSLog(@"contour_len:%f", contour_len);
    cv::erode(bin_image_copy, bin_image_copy, element, cv::Point(-1, -1), 1);
    
    std::vector<std::vector<cv::Point>> contours_main;
    findContours(bin_image_copy, contours_main, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);
    
    double contour_main_len = 0;
    for (int i = 0; i < contours_main.size(); i++) {
        contour_main_len += cv::arcLength(contours_main.at(i), 0);
    }
    NSLog(@"contour_main_len:%f", contour_main_len);
    
    //return MatToUIImage(bin_image_copy);
    
    int count = 0;
    for (int i = 0; i < contours.size(); i++) {
        // 面積の取得
        double area = cv::contourArea(contours[i], false);
        // 一定以上の面積の場合のみ
        if (area > 1000) {
            // 輪郭を近似曲線する
            std::vector<cv::Point> approx;
            cv::approxPolyDP(cv::Mat(contours[i]), approx, 0.01 * cv::arcLength(contours[i], true), true);
            
            NSLog(@"size:%ld", approx.size());
            cv::drawContours(mat, contours, i, cv::Scalar(255, 0, 0, 255), 3);
            cv::fillConvexPoly(mat, &approx[0], (int)approx.size(), cv::Scalar(255, 0, 0, 255));
            /*
            // 矩形に限定
            if (approx.size() == 4) {
                // 矩形の描画
                cv::drawContours(mat, contours, i, cv::Scalar(255, 0, 0, 255), 3);
                
                cv::Point2f src[4];
                cv::Point2f dst[4];
                
                src[0] = approx[2];
                src[1] = approx[1];
                src[2] = approx[3];
                src[3] = approx[0];
                
                //dst[0] = cv::Point2f(0.0f, 0.0f);
                //dst[1] = cv::Point2f(image2.size.width, 0.0f);
                //dst[2] = cv::Point2f(0.0f, image2.size.height);
                //dst[3] = cv::Point2f(image2.size.width, image2.size.height);
                dst[3] = cv::Point2f(0.0f, 0.0f);
                dst[2] = cv::Point2f(image.size.width, 0.0f);
                dst[1] = cv::Point2f(0.0f, image.size.height);
                dst[0] = cv::Point2f(image.size.width, image.size.height);
                
                count ++;
                //cv::Mat perspective_matrix = cv::getPerspectiveTransform(src, dst);
                //cv::warpPerspective(mat, mat, perspective_matrix, mat.size(), cv::INTER_LINEAR);
            }
             */
        }
    }
    //NSLog(@"count : %d", count);
    
    return MatToUIImage(mat);
}

- (UIImage *)lineDetect:(UIImage *)image {
    Mat mat;
    UIImageToMat(image, mat);
    
    Mat src_image = mat.clone();
    cv::GaussianBlur(src_image, src_image, cv::Size(5, 5), 0);
    
    Mat channels[4];
    cv::split(mat, channels);
    Mat canny_r;
    Mat canny_g;
    Mat canny_b;
    Mat canny_image;
    cv::Canny(channels[2], canny_r, 60.0, 180.0, 3);
    cv::Canny(channels[1], canny_g, 60.0, 180.0, 3);
    cv::Canny(channels[0], canny_b, 60.0, 180.0, 3);
    cv::bitwise_or(canny_r, canny_g, canny_image);
    cv::bitwise_or(canny_image, canny_b, canny_image);
    
    //return MatToUIImage(channels[0]);
    
    /*
    cv::Canny(mat, canny, 300, 450);
     */
    Mat lines;
    vector<Vec4i> lines_p;
    Vec4i pt;
    cv::HoughLinesP(canny_image, lines_p, 1, CV_PI / 180, 60, 40, 5);
    
    cv::Point pt1;
    cv::Point pt2;
    /*
    if (lines_p.size() > 2) {
        cv::Vec4i min_p = lines_p.at(0);
        cv::Vec4i max_p = lines_p.at(0);
        for (auto it = lines_p.begin(); it != lines_p.end(); it++) {
            pt = *it;
            if (min_p[0] > pt[0]) {
                min_p = *it;
            }
            if (max_p[0] < pt[0]) {
                max_p = *it;
            }
        }
        cv::line(mat, cv::Point(min_p[0], min_p[1]), cv::Point(min_p[2], min_p[3]), cv::Scalar(255, 0, 0, 255), 2, CV_AA);
        cv::line(mat, cv::Point(max_p[0], max_p[1]), cv::Point(max_p[2], max_p[3]), cv::Scalar(255, 0, 0, 255), 2, CV_AA);
    }
     */
    NSLog(@"size:%d", (int)lines_p.size());
    for (auto it = lines_p.begin(); it != lines_p.end(); it++) {
        pt = *it;
        cv::line(mat, cv::Point(pt[0], pt[1]), cv::Point(pt[2], pt[3]), cv::Scalar(255, 0, 0, 255), 2, CV_AA);
    }

    return MatToUIImage(mat);
}

- (void)processImage:(cv::Mat&)image {
    //Mat image_copy;
    //cvtColor(image, image_copy, CV_BGRA2BGR);
    
    //bitwise_not(image_copy, image_copy);
    //cvtColor(image_copy, image, CV_BGR2BGRA);
    
    //UIImage *img = [self binarization_otsu:MatToUIImage(image)];
    //UIImage *img = [self imageFilter:MatToUIImage(image)];
    //UIImageToMat(img, image);
    
    UIImage *img = [self lineDetect:MatToUIImage(image)];
    UIImageToMat(img, image);
}

- (void)createCameraWithParentView:(UIImageView*)parentView {
    cvCamera = [[CvVideoCamera alloc]initWithParentView:parentView];
    
    cvCamera.defaultAVCaptureDevicePosition = AVCaptureDevicePositionBack;
    cvCamera.defaultAVCaptureSessionPreset = AVCaptureSessionPreset640x480;
    cvCamera.defaultAVCaptureVideoOrientation = AVCaptureVideoOrientationPortrait;
    cvCamera.defaultFPS = 30;
    cvCamera.grayscaleMode = NO;
    
    cvCamera.delegate = self;
}

- (void)start {
    [cvCamera start];
}

@end
