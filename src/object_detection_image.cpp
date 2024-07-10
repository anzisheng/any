////./detect_object_image --model ../models/yolov8n-face_bs=1.onnx --input ../images/person.jpg
//./detect_object_image --model ./yolov8n-face.onnx  --landmarks ^Cfan4.onnx --input ../images/1.jpg
#include  "Face68Landmarks.h"
#include "facerecognizer.h"
#include "cmd_line_util.h"
#include "yolov8.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

// Runs object detection on an input image then saves the annotated image to disk.
int main(int argc, char *argv[]) {
    YoloV8Config config;
    std::string onnxModelPath;
    std::string onnxModelPathLandmark;
    std::string inputImage = "../images/6.jpg";

    // Parse the command line arguments
    // if (!parseArguments(argc, argv, config, onnxModelPath, onnxModelPathLandmark, inputImage)) {
    //     return -1;
    // }

    // Create the YoloV8 engine
    YoloV8 yoloV8("yolov8n-face.onnx", config); //
    Face68Landmarks detect_68landmarks_net("2dfan4.onnx", config);
    FaceEmbdding face_embedding_net("arcface_w600k_r50.onnx", config);
    

    // Read the input image
    cv::Mat img = cv::imread(inputImage);
    cv::Mat source_img = img.clone();
    if (img.empty()) {
        std::cout << "Error: Unable to read image at path '" << inputImage << "'" << std::endl;
        return -1;
    }

    // Run inference
    //const auto objects = yoloV8.detectObjects(img);
    std::vector<Object>objects = yoloV8.detectObjects(img);
    //Object  obj = objects[0];

    // Draw the bounding boxes on the image
    yoloV8.drawObjectLabels(img, objects);

    std::cout << "Detected " << objects.size() << " objects" << std::endl;

    //landmark 
    // Face68Landmarks  detect_68landmarks_net(onnxModelPathLandmark, config);

    // Save the image to disk
    const auto outputName = inputImage.substr(0, inputImage.find_last_of('.')) + "_annotated.jpg";
    cv::imwrite(outputName, img);
    std::cout << "Saved annotated image to: " << outputName << std::endl;

    std::vector<cv::Point2f> face_landmark_5of68;
    std::vector<cv::Point2f> face68landmarks = detect_68landmarks_net.detectlandmark(img, objects[0], face_landmark_5of68);
    std::cout << "face_landmark_5of68 size" <<face_landmark_5of68.size()<<std::endl;
    

    //face_embedding_net    
    ifstream srcFile("5landmark.txt", ios::in); 
    if(!srcFile.is_open())
    {
        cout << "cann't open 5landmark.txt"<<endl;
    }
    std::cout <<"5landmark.txt" << endl;
    for(int i = 0; i< face_landmark_5of68.size(); i++)
    {
        cout << i <<"  :";
        
        float x; 
        srcFile >> x; 
        std::cout << x <<"  ";
        face_landmark_5of68[i].x = x;
        
        float y; 
        srcFile >> y;
        std::cout << y <<"  "<<endl;
        face_landmark_5of68[i].y = y;
        
    }
    srcFile.close();
    cout << "verify face_landmark_5of68:"<<endl;
    for(int i = 0; i< face_landmark_5of68.size(); i++)
    {        
        cout << face_landmark_5of68[i].x << "   "<< face_landmark_5of68[i].y <<endl ;       
        
    }


    vector<float> source_face_embedding = face_embedding_net.detect(source_img, face_landmark_5of68);
    ofstream destFile2("embedding_cpp.txt", ios::out); 

	for(int i =0; i < source_face_embedding.size(); i++)
	{
		destFile2 << source_face_embedding[i] << " " ;
	}
	destFile2.close();

    return 0;
}