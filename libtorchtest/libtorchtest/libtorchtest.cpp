#include <torch/script.h> // One-stop header.
#include <iostream>
#include <memory>
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

int main(int argc, const char* argv[]) {
    /*if (argc != 2) {
        cerr << "usage: example-app <path-to-exported-script-module>\n";
        return -1;
    }*/

    // Deserialize the ScriptModule from a file using torch::jit::load().
    torch::jit::script::Module module = torch::jit::load("torch_script_eval.pt");

    /*
    auto image = cv::imread("test.jpg");
    //cv::Mat image_transfomed = cv::Mat::zeros(image.rows, image.cols,3);
    cv::Mat image_transfomed;
    cv::resize(image, image_transfomed, cv::Size(360, 360));
    image.convertTo(image, CV_32FC3, 1.0f / 255.f);
    //cv::imshow("frame", image);
    //cv::waitKey(30);

    // 转换为Tensor
    torch::Tensor tensor_image = torch::from_blob(image_transfomed.data, {1, image_transfomed.rows, image_transfomed.cols,3}, torch::kByte);
    tensor_image = tensor_image.permute({ 0,3,1,2 });
    tensor_image = tensor_image.toType(torch::kFloat);
    tensor_image = tensor_image.div(255);
    //tensor_image = tensor_image.unsqueeze(0);

    //cout << tensor_image << endl;


    module.eval();

    auto output = module.forward({tensor_image});
    auto maskmat = output.toTuple()->elements()[1];
    maskmat = maskmat.toTensor().squeeze(0);
    torch::Tensor a = torch::randn({ 11000,2 });
    a = maskmat.toTensor();
    auto scoremask = a > 0.5;
    //torch::Tensor index = torch::ones({ 11000, 1 });
    // auto maskscore = torch::index_select(scoremask, 1, index);
    auto maskflag = scoremask.select(1, 0);
    auto result = torch::max(maskflag, 0);
    auto finalflag = get<0>(result).toType(torch::kBool);
    //auto finalflag = get<0>(result);

    int64_t ans = 0;
    ans = finalflag.item<int64_t>();
    if (ans == 0) {
        cout << "no mask" << endl;
    }
    else
        cout << "mask" << endl;


  /*  cout << output.toTuple()->elements()[0].isTensor() << endl;
    cout << output.toTuple()->elements()[1].isTensor() << endl;*/


    //assert(module != nullptr);*/

    VideoCapture capture(0);
    while (true) {
        Mat frame;
        capture >> frame;
        auto image = frame;
        //cv::Mat image_transfomed = cv::Mat::zeros(image.rows, image.cols,3);
        cv::Mat image_transfomed;
        cv::resize(image, image_transfomed, cv::Size(360, 360));
        image.convertTo(image, CV_32FC3, 1.0f / 255.f);
        //cv::imshow("frame", image);
        //cv::waitKey(30);

        // 转换为Tensor
        torch::Tensor tensor_image = torch::from_blob(image_transfomed.data, { 1, image_transfomed.rows, image_transfomed.cols,3 }, torch::kByte);
        tensor_image = tensor_image.permute({ 0,3,1,2 });
        tensor_image = tensor_image.toType(torch::kFloat);
        tensor_image = tensor_image.div(255);
        //tensor_image = tensor_image.unsqueeze(0);

        //cout << tensor_image << endl;


        module.eval();

        auto output = module.forward({ tensor_image });
        auto maskmat = output.toTuple()->elements()[1];
        maskmat = maskmat.toTensor().squeeze(0);
        torch::Tensor a = torch::randn({ 11000,2 });
        // a = maskmat.toTensor();
        // auto scoremask = a > 0.5;
        //torch::Tensor index = torch::ones({ 11000, 1 });
        // auto maskscore = torch::index_select(scoremask, 1, index);
        // auto temmaskscore = scoremask * a;
        // auto maskflag = scoremask.select(1, 0);
        auto score = torch::max(maskmat.toTensor(), 1);
        // cout << get<0>(score) << endl;
        //auto maskflag = temmaskscore.select(1, 0);
        //auto result = torch::max(maskflag, 0);
        auto maxscore = torch::max(get<0>(score), 0);
        auto finalscore = get<0>(maxscore).toType(torch::kFloat16);
        auto finalscore_index = get<1>(maxscore).toType(torch::kI64);
        // cout << finalscore << "," << finalscore_index << endl;
        int64_t index = 0;
        index = finalscore_index.item<int64_t>();
        cout << index << endl;
        auto finalbox = maskmat.toTensor()[index];
        // cout << finalbox << endl;
        // cout << finalbox.sizes()[0] << endl;
        bool flag = false;
        for (int i = 0; i < finalbox.sizes()[0]; i++) {
            //cout << finalbox[i] << "," << finalscore << endl;
            //cout << i << "," << (finalbox[i].item<float>() > 0.5) << "," << (finalscore.item<float>() == finalbox[i].item<float>()) << endl;
            if (finalbox[i].item<float>() > 0.5) {
                if (abs(finalscore.item<float>() - finalbox[i].item<float>() <= 0.01) && i == 0) { flag = true; break; }
            }
        }
        if(flag)
            cout << "mask" << endl;
        else
            cout << "no mask" << endl;
        //auto finalflag = get<0>(result).toType(torch::kBool);
        ////auto finalflag = get<0>(result);

        //int64_t ans = 0;
        //ans = finalflag.item<int64_t>();
        //auto finalflag = get<0>(result).toType(torch::kFloat16);
        //float ans = 0.0;
        ////ans = finalflag.item<float>();
        //if (ans == 0.0) {
        //    cout << "no mask" << endl;
        //}
        //else
        //    cout << "mask" << endl;
        imshow("读取视频", frame);
        waitKey(30);	//延时30	
    }

    cout << "ok\n";

    system("pause");
    return 0;
}