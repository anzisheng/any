#include "faceswap.h"
#include "utile.h"
#include <vector>
using namespace std;

SwapFace::SwapFace(std::string onnxModelPath, const YoloV8Config &config)
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

    cout << "Create our TensorRT inference engine begin"<<endl;
    // Create our TensorRT inference engine
    m_trtEngine_swap = std::make_unique<Engine<float>>(options);

    // Build the onnx model into a TensorRT engine file, cache the file to disk, and then load the TensorRT engine file into memory.
    // If the engine file already exists on disk, this function will not rebuild but only load into memory.
    // The engine file is rebuilt any time the above Options are changed.
    auto succ = m_trtEngine_swap->buildLoadNetwork(onnxModelPath, SUB_VALS, DIV_VALS, NORMALIZE);
    if (!succ) {
        const std::string errMsg = "Error: Unable to build or load the TensorRT engine. "
                                   "Try increasing TensorRT log severity to kVERBOSE (in /libs/tensorrt-cpp-api/engine.cpp).";
        throw std::runtime_error(errMsg);
    }
    cout << "Create our TensorRT inference engine end"<<endl;
    const auto &inputDims =  m_trtEngine_swap->getInputDims();
    cout << "m_trtEngine_swap input:"<< inputDims.size()<<endl;
    //cout << inputDims.size()<<endl;

    const int length = this->len_feature*this->len_feature;
    this->model_matrix = new float[length];
    cout<<"start read model_matrix.bin"<<endl;
    FILE* fp = fopen("./model_matrix.bin", "rb");
    size_t ret = fread(this->model_matrix, sizeof(float), length, fp);//导入数据
    if (ret) {}
    fclose(fp);//关闭文件
    cout<<"read model_matrix.bin finish"<<endl;


    ////在这里就直接定义了，没有像python程序里的那样normed_template = TEMPLATES.get(template) * crop_size
    this->normed_template.emplace_back(Point2f(46.29459968, 51.69629952));
    this->normed_template.emplace_back(Point2f(81.53180032, 51.50140032));
    this->normed_template.emplace_back(Point2f(64.02519936, 71.73660032));
    this->normed_template.emplace_back(Point2f(49.54930048, 92.36550016));
    this->normed_template.emplace_back(Point2f(78.72989952, 92.20409984));
   
}

std::vector<std::vector<cv::cuda::GpuMat>> SwapFace::preprocess(Mat srcimg, const vector<Point2f> face_landmark_5, const vector<float> source_face_embedding, Mat& affine_matrix, Mat& box_mask)
{
    Mat crop_img;
    affine_matrix = warp_face_by_face_landmark_5(srcimg, crop_img, face_landmark_5, this->normed_template, Size(128, 128));
    cout <<"affine_matrix:\n";
    cout << affine_matrix<<endl;
    cv::cuda::GpuMat gpu_affine_matrix;
    gpu_affine_matrix.upload(affine_matrix);

    const int crop_size[2] = {crop_img.cols, crop_img.rows};
    box_mask = create_static_box_mask(crop_size, this->FACE_MASK_BLUR, this->FACE_MASK_PADDING);
    cout <<"box_mask:\n";
    cout << crop_img.cols<< "   "<< crop_img.rows<<endl;
    //cout << box_mask<<endl;
    cv::cuda::GpuMat gpu_box_mask;
    gpu_box_mask.upload(box_mask);

    //convertTo loss
    crop_img.convertTo(crop_img, CV_32FC1, 1 / 255.0, 0); 
    // ?? srcImg or crop_img

    cv::cuda::GpuMat gpuImg;
    gpuImg.upload(crop_img);
    //cv::cuda::cvtColor(gpuImg, gpuImg, cv::COLOR_BGR2RGB);
    cv::cuda::GpuMat rgbMat;
    cv::cuda::cvtColor(gpuImg, rgbMat, cv::COLOR_BGR2RGB);

    
    float linalg_norm = 0;
    for(int i=0;i<this->len_feature;i++)
    {
        linalg_norm += powf(source_face_embedding[i], 2);
    }
    linalg_norm = sqrt(linalg_norm);
    this->input_embedding.resize(this->len_feature);
    for(int i=0;i<this->len_feature;i++)
    {
        float sum=0;
        for(int j=0;j<this->len_feature;j++)
        {
            sum += (source_face_embedding[j]*this->model_matrix[j*this->len_feature+i]);
        }
        this->input_embedding[i] = sum/linalg_norm;
    }
    /**
    int nbChannels=48;
    cv::Mat image2(cv::Size(480,640), CV_32FC(nbChannels))
     */
     int nbChannels=512;
    //cv::cuda::GpuMat gpu_input_embedding =  cv::cuda::GpuMat(0,0, CV_32FC(nbChannels));///(int)this->input_embedding.size());
    //input_embedding.setchannel(512)<<endl;

    //input_embedding = input_embedding.reshape(0, 512);
    //gpu_input_embedding.upload(this->input_embedding);
    cv::Mat m(0,0, CV_8UC(nbChannels)); 
    cout << "embedding"<< m.channels()<<endl;

    //m.reshape(512, 0) ;
    cv::cuda::GpuMat gpu_input_embedding =  cv::cuda::GpuMat(m);
    cout << "gpu embedding"<< gpu_input_embedding.channels()<<endl;

    cv::cuda::GpuMat gpu_input_embedding2 = gpu_input_embedding.reshape(512);
    cout << "gpu embedding2"<< gpu_input_embedding2.channels()<<endl;

    cout << "begin to pack the inputs"<<endl;

    auto resized = rgbMat;
    // Populate the input vectors
    const auto &inputDims = m_trtEngine_swap->getInputDims();

    // Resize to the model expected input size while maintaining the aspect ratio with the use of padding
    if (resized.rows != inputDims[0].d[1] || resized.cols != inputDims[0].d[2]) {
        // Only resize if not already the right size to avoid unecessary copy
        resized = Engine<float>::resizeKeepAspectRatioPadRightBottom(rgbMat, inputDims[0].d[1], inputDims[0].d[2]);
    }

    // Convert to format expected by our inference engine
    // The reason for the strange format is because it supports models with multiple inputs as well as batching
    // In our case though, the model only has a single input and we are using a batch size of 1.
    std::vector<cv::cuda::GpuMat> input{std::move(resized)};
    std::vector<cv::cuda::GpuMat> input2{std::move(gpu_input_embedding2)};
    //input.push_back(std::move(gpu_input_embedding));
    cout << "inputs 2 gpuMat"<<endl;
    std::vector<std::vector<cv::cuda::GpuMat>> inputs{std::move(input)};
    inputs.push_back(std::move(input2));
    cout << "inputs has one input which has two gpumat"<<endl;


    // These params will be used in the post-processing stage
    m_imgHeight = rgbMat.rows;
    m_imgWidth = rgbMat.cols;
    m_ratio = 1.f / std::min(inputDims[0].d[2] / static_cast<float>(rgbMat.cols), inputDims[0].d[1] / static_cast<float>(rgbMat.rows));

    return inputs;
    

}

cv::Mat SwapFace::process(cv::Mat target_img, const std::vector<float> source_face_embedding, const std::vector<cv::Point2f> target_landmark_5)
{
    Mat affine_matrix;
    Mat box_mask;
    //this->preprocess(target_img, target_landmark_5, source_face_embedding, affine_matrix, box_mask);
    //preprocess
    const auto input =this->preprocess(target_img, target_landmark_5, source_face_embedding, affine_matrix, box_mask);   
    
    //runInference
    std::vector<std::vector<std::vector<float>>> featureVectors;
    auto succ = m_trtEngine_swap->runInference_2v1(input, featureVectors);

    const auto &numOutputs = m_trtEngine_swap->getOutputDims().size();

      
    //preprocess
    
    std::vector<float> featureVector;
    Engine<float>::transformOutput(featureVectors, featureVector);
    cv::Mat mat = postprocess(featureVector);
    return mat;
    
}

/*
cv::Mat SwapFace::process(cv::cuda::GpuMat& gpuImg, const std::vector<float> source_face_embedding, const std::vector<cv::Point2f> target_landmark_5)
{

    


    //runinference


    //postprocesss


    //return result

}
*/

cv::Mat SwapFace::postprocess(std::vector<float> &featureVector) {

}