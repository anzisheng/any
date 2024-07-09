#include "Face68Landmarks.h"
#include "engine.h"
using namespace std;
using namespace cv;

Face68Landmarks::Face68Landmarks(const std::string &onnxModelPath, const YoloV8Config &config)
{
    
    
    // Specify options for GPU inference
    Options options;
    options.optBatchSize = 1;
    options.maxBatchSize = 1;

    options.precision = config.precision;
    options.calibrationDataDirectoryPath = config.calibrationDataDirectory;

    if (options.precision == Precision::INT8) {
        if (options.calibrationDataDirectoryPath.empty()) {
            throw std::runtime_error("Error: Must supply calibration data path for INT8 calibration");
        }
    }  
    
    
        // Create our TensorRT inference engine
    m_trtEngine_landmark = std::make_unique<Engine<float>>(options);

    // Build the onnx model into a TensorRT engine file, cache the file to disk, and then load the TensorRT engine file into memory.
    // If the engine file already exists on disk, this function will not rebuild but only load into memory.
    // The engine file is rebuilt any time the above Options are changed.
    auto succ = m_trtEngine_landmark->buildLoadNetwork(onnxModelPath, SUB_VALS, DIV_VALS, NORMALIZE);
    if (!succ) {
        const std::string errMsg = "Error: Unable to build or load the TensorRT engine. "
                                   "Try increasing TensorRT log severity to kVERBOSE (in /libs/tensorrt-cpp-api/engine.cpp).";
        throw std::runtime_error(errMsg);
    }
}

/*

//std::vector<std::vector<cv::cuda::GpuMat>> Face68Landmarks::preprocess(const cv::cuda::GpuMat &gpuImg);
std::vector<std::vector<cv::cuda::GpuMat>> Face68Landmarks::preprocess(const cv::cuda::GpuMat &gpuImg) {
    // Populate the input vectors
    const auto &inputDims = m_trtEngine_landmark->getInputDims();

    // Convert the image from BGR to RGB
    cv::cuda::GpuMat rgbMat;
    cv::cuda::cvtColor(gpuImg, rgbMat, cv::COLOR_BGR2RGB);

    auto resized = rgbMat;

    // Resize to the model expected input size while maintaining the aspect ratio with the use of padding
    if (resized.rows != inputDims[0].d[1] || resized.cols != inputDims[0].d[2]) {
        // Only resize if not already the right size to avoid unecessary copy
        resized = Engine<float>::resizeKeepAspectRatioPadRightBottom(rgbMat, inputDims[0].d[1], inputDims[0].d[2]);
    }

    // Convert to format expected by our inference engine
    // The reason for the strange format is because it supports models with multiple inputs as well as batching
    // In our case though, the model only has a single input and we are using a batch size of 1.
    std::vector<cv::cuda::GpuMat> input{std::move(resized)};
    std::vector<std::vector<cv::cuda::GpuMat>> inputs{std::move(input)};

    // These params will be used in the post-processing stage
    m_imgHeight = rgbMat.rows;
    m_imgWidth = rgbMat.cols;
    m_ratio = 1.f / std::min(inputDims[0].d[2] / static_cast<float>(rgbMat.cols), inputDims[0].d[1] / static_cast<float>(rgbMat.rows));

    return inputs;
}
*/
std::vector<std::vector<cv::cuda::GpuMat>> Face68Landmarks::preprocess(const cv::cuda::GpuMat &gpuImg) {
    // Populate the input vectors
    const auto &inputDims = m_trtEngine_landmark->getInputDims();
    //std::cout << "inputDims" <<inputDims.dims <<std::endl;

    // Convert the image from BGR to RGB
    cv::cuda::GpuMat rgbMat;
    cv::cuda::cvtColor(gpuImg, rgbMat, cv::COLOR_BGR2RGB);

    auto resized = rgbMat;

    // Resize to the model expected input size while maintaining the aspect ratio with the use of padding
    std::cout << inputDims[0].d[1] << "   " << inputDims[0].d[2] << std::endl;
    if (resized.rows != inputDims[0].d[1] || resized.cols != inputDims[0].d[2]) {
        // Only resize if not already the right size to avoid unecessary copy
        resized = Engine<float>::resizeKeepAspectRatioPadRightBottom(rgbMat, inputDims[0].d[1], inputDims[0].d[2]);
    }
    std::cout <<"resized: "<< resized.rows << "   " << resized.cols << std::endl;

    // Convert to format expected by our inference engine
    // The reason for the strange format is because it supports models with multiple inputs as well as batching
    // In our case though, the model only has a single input and we are using a batch size of 1.
    std::vector<cv::cuda::GpuMat> input{std::move(resized)};
    std::vector<std::vector<cv::cuda::GpuMat>> inputs{std::move(input)};

    // These params will be used in the post-processing stage
    m_imgHeight = rgbMat.rows;
    m_imgWidth = rgbMat.cols;
    m_ratio = 1.f / std::min(inputDims[0].d[2] / static_cast<float>(rgbMat.cols), inputDims[0].d[1] / static_cast<float>(rgbMat.rows));
    std::cout << "inputs size:"<< inputs.size()<<std::endl;
    return inputs;
}


std::vector<cv::Point2f> Face68Landmarks::postprocess(std::vector<float> &featureVector)
{
    std::cout <<"featureVector: " << featureVector.size()<<std::endl;
    std::vector<Point2f> ret;
    const int num_points = featureVector.size() / 3; //3 represent x, y and score
    float *pdata  = featureVector.data();
    vector<Point2f> face_landmark_68(num_points);
    std::cout << "befor transform \n";
    for (int i = 0; i < num_points; i++)
    {
        float x = pdata[i * 3] / 64.0 * 256.0;
        float y = pdata[i * 3 + 1] / 64.0 * 256.0;
        face_landmark_68[i] = Point2f(x, y);
        cout <<i <<": "<< x <<"   "<<y <<std::endl;
        circle(m_srcImg, face_landmark_68[i], 3 ,Scalar(0,255,0),-1);
    }
    imwrite("landmark.jpg",m_srcImg);


    return ret;
}

    
vector<Point2f> Face68Landmarks::detectlandmark(const cv::cuda::GpuMat &inputImageBGR, Object& object,vector<Point2f> &face_landmark_5of68)
    {
        //preprocess, get input
        const auto input = preprocess(inputImageBGR);

        //send to network
        std::vector<std::vector<std::vector<float>>> featureVectors;
        auto succ = m_trtEngine_landmark->runInference(input, featureVectors);

        if (!succ) {
             throw std::runtime_error("Error: Unable to run inference.");
         }
         else
         {
            std::cout << "land mark network is over"<<std::endl;
         }
         
        // Check if our model does only object detection or also supports segmentation
        std::vector<Object> ret;
        const auto &numOutputs = m_trtEngine_landmark->getOutputDims().size();
        std::cout<<"numOutputs size " << numOutputs<< std::endl;
        if(numOutputs == 2)
        {
            // Since we have a batch size of 1 and 2 outputs, we must convert the output from a 3D array to a 2D array.
            std::vector<std::vector<float>> featureVector;
            Engine<float>::transformOutput(featureVectors, featureVector);
            cout<<"1st dim size"<< featureVector.size() <<std::endl; 
            for(int i = 1; i < featureVector.size(); i++)
            {
                cout << "ith size " <<featureVector[i].size()<<std::endl;

                for(int j = 0; j < featureVector[i].size(); j++)
                    cout <<  featureVector[i][j] << " ";
                cout << std::endl;
            }
        std::vector<Point2f> result;
        result = postprocess(featureVector[1]);




        }

        
        


        //std::vector<float> featureVector;
        //Engine<float>::transformOutput(featureVectors, featureVector);

        //std::cout<<"featureVector size" << featureVector.data()<< std::endl;



        //get the network output  const std::vector<nvinfer1::Dims>
        //const auto &numOutputs = m_trtEngine_landmark->getOutputDims().size();
        //std::cout<<"output:" <<((std::vector<nvinfer1::Dims>)numOutputs).size()<<std::endl;
        //const int num_points = ort_outputs[0].GetTensorTypeAndShapeInfo().GetShape()[1];
        //const int num_points = numOutputs[0].GetTensorTypeAndShapeInfo().GetShape()[1];
        //const int num_points = ((std::vector<nvinfer1::Dims>)numOutputs)[0];
        //std::cout << "num_points:"<<num_points << std::endl;

        
        // float *pdata = ort_outputs[0].GetTensorMutableData<float>(); /// 形状是(1, 68, 3), 每一行的长度是3，表示一个关键点坐标x,y和置信度
        // const int num_points = ort_outputs[0].GetTensorTypeAndShapeInfo().GetShape()[1];

        // vector<Point2f> face_landmark_68(num_points);
        // for (int i = 0; i < num_points; i++)
        // {
        //     float x = pdata[i * 3] / 64.0 * 256.0;
        //     float y = pdata[i * 3 + 1] / 64.0 * 256.0;
        //     face_landmark_68[i] = Point2f(x, y);
        // }
        vector<Point2f> face68landmarks;
        // cv::transform(face_landmark_68, face68landmarks, this->inv_affine_matrix);

        return face68landmarks;
        

    }


vector<Point2f> Face68Landmarks::detectlandmark(const cv::Mat &inputImageBGR,Object& object, vector<Point2f> &face_landmark_5of68){
    
    m_srcImg = inputImageBGR;
    // Upload the image to GPU memory
    cv::cuda::GpuMat gpuImg;
    gpuImg.upload(inputImageBGR);

    std::cout << object.rect.x <<"  "<<object.rect.y <<std::endl;

    
    // Call detectObjects with the GPU image
    return detectlandmark(gpuImg, object, face_landmark_5of68);
}



//}
