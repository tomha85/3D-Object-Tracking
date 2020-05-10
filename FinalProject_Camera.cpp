#include<iostream>
#include<fstream>
#include<sstream>
#include<iomanip>
#include<vector>
#include<cmath>
#include<limits>
#include<opencv2/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/features2d.hpp>
#include<opencv2/xfeatures2d.hpp>
#include<opencv2/xfeatures2d/nonfree.hpp>

#include "dataStructures.h"
#include "matching2D.hpp"
#include "objectDetection2D.hpp"
#include "lidarData.hpp"
#include "camFusion.hpp"
using namespace std;

int main(int argc, const char *argv[])
{
    string dataPath = "../";
    string imgBasePath = dataPath + "images/";
    string imgPrefix = "KITTI/2011_09_26/image_02/data/000000"; 
    string imgFileType = ".png";
    int imgStartIndex = 0; 
    int imgEndIndex = 18;   
    int imgStepWidth = 1; 
    int imgFillWidth = 4;  
    

    string yoloBasePath = dataPath + "dat/yolo/";
    string yoloClassesFile = yoloBasePath + "coco.names";   
    string yoloModelConfiguration = yoloBasePath + "yolov3.cfg";
    string yoloModelWeights = yoloBasePath + "yolov3.weights";
    string lidarPrefix = "KITTI/2011_09_26/velodyne_points/data/000000";
    string lidarFileType = ".bin";

    cv::Mat P_rect_00(3,4,cv::DataType<double>::type); 
    cv::Mat R_rect_00(4,4,cv::DataType<double>::type); 
    cv::Mat RT(4,4,cv::DataType<double>::type); 
    
    RT.at<double>(0,0) = 7.533745e-03; RT.at<double>(0,1) = -9.999714e-01; RT.at<double>(0,2) = -6.166020e-04; RT.at<double>(0,3) = -4.069766e-03;
    RT.at<double>(1,0) = 1.480249e-02; RT.at<double>(1,1) = 7.280733e-04; RT.at<double>(1,2) = -9.998902e-01; RT.at<double>(1,3) = -7.631618e-02;
    RT.at<double>(2,0) = 9.998621e-01; RT.at<double>(2,1) = 7.523790e-03; RT.at<double>(2,2) = 1.480755e-02; RT.at<double>(2,3) = -2.717806e-01;
    RT.at<double>(3,0) = 0.0; RT.at<double>(3,1) = 0.0; RT.at<double>(3,2) = 0.0; RT.at<double>(3,3) = 1.0;
    
    R_rect_00.at<double>(0,0) = 9.999239e-01; R_rect_00.at<double>(0,1) = 9.837760e-03; R_rect_00.at<double>(0,2) = -7.445048e-03; R_rect_00.at<double>(0,3) = 0.0;
    R_rect_00.at<double>(1,0) = -9.869795e-03; R_rect_00.at<double>(1,1) = 9.999421e-01; R_rect_00.at<double>(1,2) = -4.278459e-03; R_rect_00.at<double>(1,3) = 0.0;
    R_rect_00.at<double>(2,0) = 7.402527e-03; R_rect_00.at<double>(2,1) = 4.351614e-03; R_rect_00.at<double>(2,2) = 9.999631e-01; R_rect_00.at<double>(2,3) = 0.0;
    R_rect_00.at<double>(3,0) = 0; R_rect_00.at<double>(3,1) = 0; R_rect_00.at<double>(3,2) = 0; R_rect_00.at<double>(3,3) = 1;
    
    P_rect_00.at<double>(0,0) = 7.215377e+02; P_rect_00.at<double>(0,1) = 0.000000e+00; P_rect_00.at<double>(0,2) = 6.095593e+02; P_rect_00.at<double>(0,3) = 0.000000e+00;
    P_rect_00.at<double>(1,0) = 0.000000e+00; P_rect_00.at<double>(1,1) = 7.215377e+02; P_rect_00.at<double>(1,2) = 1.728540e+02; P_rect_00.at<double>(1,3) = 0.000000e+00;
    P_rect_00.at<double>(2,0) = 0.000000e+00; P_rect_00.at<double>(2,1) = 0.000000e+00; P_rect_00.at<double>(2,2) = 1.000000e+00; P_rect_00.at<double>(2,3) = 0.000000e+00;    
    
    double sensorFrameRate = 10.0 / imgStepWidth; 
    int dataBufferSize = 2;      
    vector<DataFrame> dataBuffer; 
    bool bVis = false;                
    ofstream matches_file;
    matches_file.open ("../Matches_keypoint.csv");
    matches_file << "Detector, Descriptor, Img#, TTC_Lidar, TTC_Camera"  << endl;
     
	std::vector<std::string> detector_type_names = { "SHITOMASI", "FAST", "BRISK", "ORB", "AKAZE"};
    std::vector<std::string> descriptor_type_names = {"BRISK", "BRIEF", "ORB", "FREAK"};
    for(auto detector_type_name:detector_type_names) 
    {
        for(auto descriptor_type_name:descriptor_type_names) 
        {
            dataBuffer.clear();                  
            
            for (size_t imgIndex = 0; imgIndex <= imgEndIndex - imgStartIndex; imgIndex+=imgStepWidth)
            {       
                ostringstream imgNumber;
                imgNumber << setfill('0') << setw(imgFillWidth) << imgStartIndex + imgIndex;
                string imgFullFilename = imgBasePath + imgPrefix + imgNumber.str() + imgFileType;
        
                cv::Mat img = cv::imread(imgFullFilename);

                DataFrame frame;
                frame.cameraImg = img;

                if (  dataBuffer.size() > dataBufferSize)
                {
                    dataBuffer.erase(dataBuffer.begin());
                }
                dataBuffer.push_back(frame);

                cout << "#1 :Image loading done" << endl;
        
      	        matches_file << detector_type_name << ", " << descriptor_type_name << ", " << imgNumber.str() << ", ";
        
                float confThreshold = 0.2;
                float nmsThreshold = 0.4;        
                detectObjects((dataBuffer.end() - 1)->cameraImg, (dataBuffer.end() - 1)->boundingBoxes, confThreshold, nmsThreshold,
                            yoloBasePath, yoloClassesFile, yoloModelConfiguration, yoloModelWeights, bVis);

                cout << "#2 : Classify and detect done" << endl;
        
                string lidarFullFilename = imgBasePath + lidarPrefix + imgNumber.str() + lidarFileType;
                std::vector<LidarPoint> lidarPoints;
                loadLidarFromFile(lidarPoints, lidarFullFilename);
        
                float minZ = -1.5, maxZ = -0.9, minX = 2.0, maxX = 20.0, maxY = 2.0, minR = 0.1; // focus on ego lane
                cropLidarPoints(lidarPoints, minX, maxX, maxY, minZ, maxZ, minR);
    
                (dataBuffer.end() - 1)->lidarPoints = lidarPoints;

                cout << "#3 :Crop lidarpoints done" << endl;
        
                float shrinkFactor = 0.10; 
                clusterLidarWithROI((dataBuffer.end()-1)->boundingBoxes, (dataBuffer.end() - 1)->lidarPoints, shrinkFactor, P_rect_00, R_rect_00, RT);
                bVis = false;
                if(bVis)
                {
                    show3DObjects((dataBuffer.end()-1)->boundingBoxes, cv::Size(4.0, 20.0), cv::Size(2000, 2000), true);
                }
                bVis = false;

                cout << "#4: Clustering cloud done" << endl;       
        
                cv::Mat imgGray;
                cv::cvtColor((dataBuffer.end()-1)->cameraImg, imgGray, cv::COLOR_BGR2GRAY);

        
                vector<cv::KeyPoint> keypoints;       
                string detectorType = detector_type_name; 
        
                if (detectorType.compare("SHITOMASI") == 0)
                {
                    detKeypointsShiTomasi(keypoints, imgGray, false);
                }
        
                else if (detectorType.compare("HARRIS") == 0)
                {
                    detKeypointsHarris(keypoints, imgGray, false);
                }        
                else if (detectorType.compare("FAST")  == 0 ||detectorType.compare("BRISK") == 0 ||
                        detectorType.compare("ORB")   == 0 ||detectorType.compare("AKAZE") == 0 ||
                        detectorType.compare("SIFT")  == 0)
                {
                    detKeypointsModern(keypoints, imgGray, detectorType, false);
                }
                else
                {
                    throw invalid_argument(detectorType + " is not a valid detectorType. Try SHITOMASI, HARRIS, FAST, BRISK, ORB, AKAZE, SIFT.");
                }
        
                bool bLimitKpts = false;
                if (bLimitKpts)
                {
                    int maxKeypoints = 50;

                    if (detectorType.compare("SHITOMASI") == 0)
                    { 
                        keypoints.erase(keypoints.begin() + maxKeypoints, keypoints.end());
                    }
                    cv::KeyPointsFilter::retainBest(keypoints, maxKeypoints);
                    cout << " NOTE: Keypoints have been limited!" << endl;
                }

                (dataBuffer.end() - 1)->keypoints = keypoints;

                cout << "#5 :Detect keypoints done" << endl;
        
                cv::Mat descriptors;        
                string descriptorType = descriptor_type_name;
                descKeypoints((dataBuffer.end() - 1)->keypoints, (dataBuffer.end() - 1)->cameraImg, descriptors, descriptorType);
                (dataBuffer.end() - 1)->descriptors = descriptors;

                cout << "#6 :Extractor descriptors done" << endl;

                if (dataBuffer.size() > 1)
                {
                    vector<cv::DMatch> matches;
                    string matcherType = "MAT_BF";       
                    string descriptorType = "DES_BINARY"; 
                    string selectorType = "SEL_NN";       

                    matchDescriptors((dataBuffer.end() - 2)->keypoints, (dataBuffer.end() - 1)->keypoints,
                                    (dataBuffer.end() - 2)->descriptors, (dataBuffer.end() - 1)->descriptors,
                                     matches, descriptorType, matcherType, selectorType);

            
                    (dataBuffer.end() - 1)->kptMatches = matches;

                    cout << "#7 : Matching descriptors done" << endl;
            
            
                    map<int, int> bbBestMatches;
                    matchBoundingBoxes(matches, bbBestMatches, *(dataBuffer.end()-2), *(dataBuffer.end()-1)); // associate bounding 
                    (dataBuffer.end()-1)->bbMatches = bbBestMatches;

                    cout << "#8 :Object tracking done" << endl;


                    for (auto it1 = (dataBuffer.end() - 1)->bbMatches.begin(); it1 != (dataBuffer.end() - 1)->bbMatches.end(); ++it1)
                    {                
                        BoundingBox *prevBB, *currBB;
                        for (auto it2 = (dataBuffer.end() - 1)->boundingBoxes.begin(); it2 != (dataBuffer.end() - 1)->boundingBoxes.end(); ++it2)
                        {
                            if (it1->second == it2->boxID) 
                            {
                                currBB = &(*it2);
                            }
                        }

                        for (auto it2 = (dataBuffer.end() - 2)->boundingBoxes.begin(); it2 != (dataBuffer.end() - 2)->boundingBoxes.end(); ++it2)
                        {
                            if (it1->first == it2->boxID) 
                            {
                                prevBB = &(*it2);
                            }
                        }

                        if( currBB->lidarPoints.size()>0 && prevBB->lidarPoints.size()>0 ) 
                        {                    
                            double ttcLidar; 
                            computeTTCLidar(prevBB->lidarPoints, currBB->lidarPoints, sensorFrameRate, ttcLidar);
                   
                            double ttcCamera;
                            clusterKptMatchesWithROI(*currBB, (dataBuffer.end() - 2)->keypoints, (dataBuffer.end() - 1)->keypoints, (dataBuffer.end() - 1)->kptMatches);                    
                            computeTTCCamera((dataBuffer.end() - 2)->keypoints, (dataBuffer.end() - 1)->keypoints, currBB->kptMatches, sensorFrameRate, ttcCamera);
                    
                            matches_file << ttcLidar << ", " << ttcCamera;
        
                            bVis = false;//true
                            if (bVis)
                            {
                                cv::Mat visImg = (dataBuffer.end() - 1)->cameraImg.clone();
                                showLidarImgOverlay(visImg, currBB->lidarPoints, P_rect_00, R_rect_00, RT, &visImg);
                                cv::rectangle(visImg, cv::Point(currBB->roi.x, currBB->roi.y), cv::Point(currBB->roi.x + currBB->roi.width, currBB->roi.y + currBB->roi.height), cv::Scalar(0, 255, 0), 2);
                        
                                char str[200];
                                sprintf(str, "TTC Lidar : %f s, TTC Camera : %f s", ttcLidar, ttcCamera);
                                putText(visImg, str, cv::Point2f(80, 50), cv::FONT_HERSHEY_PLAIN, 2, cv::Scalar(0,0,255));

                                string windowName = "Final Results : TTC";
                                cv::namedWindow(windowName, 4);
                                cv::imshow(windowName, visImg);
                                cout << "Press key to continue to next frame" << endl;
                                
                                cv::waitKey(0);
                            }
                            bVis = false;                           
                        } 
                    }             

                }        
                matches_file << endl;       
            }    
            matches_file << endl;
        }
    }
    matches_file.close();
    return 0;
}

//Looking at Matches_keypoints.csv file, we see top 3 best results detector/descriptor:
//1: SHITIMASI/FREAK
//2: AKAZE/BRISK
//3: AKAZE/FREAK
//ORB detector gives unrealible camera TTC estimates













