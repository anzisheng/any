////./detect_object_image --model ../models/yolov8n-face_bs=1.onnx --input ../images/person.jpg
//./detect_object_image --model ./yolov8n-face.onnx  --landmarks ^Cfan4.onnx --input ../images/1.jpg
#include  "Face68Landmarks.h"
#include "cmd_line_util.h"
#include "yolov8.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

// Runs object detection on an input image then saves the annotated image to disk.
int main(int argc, char *argv[]) {
    YoloV8Config config;
    std::string onnxModelPath;
    std::string onnxModelPathLandmark;
    std::string inputImage = "../images/1.jpg";

    // Parse the command line arguments
    // if (!parseArguments(argc, argv, config, onnxModelPath, onnxModelPathLandmark, inputImage)) {
    //     return -1;
    // }

    // Create the YoloV8 engine
    YoloV8 yoloV8("yolov8n-face.onnx", config); //
    Face68Landmarks detect_68landmarks_net("2dfan4.onnx", config);


    // Read the input image
    auto img = cv::imread(inputImage);
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

    return 0;
}