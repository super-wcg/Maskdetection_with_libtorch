#include "SRdeal.h"


//图片加载函数
Mat CSRdeal::LoadImg(std::string path)
{
	cv::Mat image;
	image = imread(path);
	//showImage(image);
	return image;
}

//图片显示函数
void CSRdeal::showImage(Mat image)
{
	namedWindow("Display window", CV_WINDOW_AUTOSIZE);// Create a window for display.
	imshow("Display window", image);
	waitKey(0);
}

//模型加载函数
torch::jit::script::Module CSRdeal::LoadModule(std::string ModelPath, bool &bFlag)
{
	torch::jit::script::Module module;
	try
	{
		module.to(at::kCUDA);
		module = torch::jit::load(ModelPath);
		bFlag = true;
		return module;
	}
	catch (const c10::Error & e)
	{
		bFlag = false;
		return module;
	}
}

//输入为两个张量的超分处理函数输出为彩色的图-中心算法处理
Mat CSRdeal::CASIASRDealImg(Mat image, torch::jit::script::Module module, bool& bFlag)
{
	cv::Mat resultImg;
	try
	{
		//输入图像转换成张量
		image.convertTo(image, CV_32FC3, 1.0f / 255.f);
		auto input_tensor = torch::from_blob(image.data, { 1, image.rows, image.cols, image.channels() });
		
		//定义与输入的同维张量
		at::Tensor outImage = torch::zeros({ 1, image.rows, image.cols, image.channels() });
		input_tensor = input_tensor.permute({ 0, 3, 1, 2 });
		outImage = outImage.permute({ 0, 3, 1, 2 });

		//进行超分处理
		module.to(at::kCPU);
		module.eval();	
		at::Tensor result = module.forward({ input_tensor, outImage }).toTensor();

		//对输出结果进行处理
		result = result.to(torch::kCPU);
		result = result.squeeze().detach().permute({ 1, 2, 0 });
		result = result.mul(255).clamp(0, 255).to(torch::kU8);

		int height = result.size(0);
		int width = result.size(1);
		int channel = result.size(2);

		//初始化新建图像
		resultImg = Mat(height, width, CV_8UC3);
		std::memcpy((void*)resultImg.data, result.data_ptr(), sizeof(torch::kU8) * result.numel());
		bFlag = true;
		return resultImg;
	}
	catch (const c10::Error & e)
	{
		bFlag = false;
		return resultImg;
	}
}

//输入为一个张量的超分处理函数输出为彩色的图-ResNet超分处理算法
Mat CSRdeal::ResNetDealImg(Mat image, torch::jit::script::Module module, bool& bFlag)
{
	cv::Mat resultImg;
	try
	{
		//输入图像转换成张量
		image.convertTo(image, CV_32FC3, 1.0f / 255.f);
		auto input_tensor = torch::from_blob(image.data, { 1, image.rows, image.cols, image.channels() });
		input_tensor = input_tensor.permute({ 0, 3, 1, 2 });

		//进行超分处理
		module.to(at::kCPU);
		module.eval();
		at::Tensor result = module.forward({ input_tensor }).toTensor();

		//对输出结果进行处理
		result = result.to(torch::kCPU);
		result = result.squeeze().detach().permute({ 1, 2, 0 });
		result = result.mul(255).clamp(0, 255).to(torch::kU8);

		int height = result.size(0);
		int width = result.size(1);
		int channel = result.size(2);

		//初始化新建图像
		resultImg = Mat(height, width, CV_8UC3);
		std::memcpy((void*)resultImg.data, result.data_ptr(), sizeof(torch::kU8) * result.numel());
		bFlag = true;
		return resultImg;
	}
	catch (const c10::Error & e)
	{
		bFlag = false;
		return resultImg;
	}
}

//输入为一个张量的超分处理函数输出为灰色的图-DensenNet超分处理算法
Mat CSRdeal::DenseNetDealImg(Mat image, torch::jit::script::Module module, bool& bFlag)
{
	cv::Mat resultImg;
	try
	{
		//输入图像转换成张量
		image.convertTo(image, CV_32FC3, 1.0f / 255.f);
		auto input_tensor = torch::from_blob(image.data, { 1, image.rows, image.cols, image.channels() });
		input_tensor = input_tensor.permute({ 0, 3, 1, 2 });

		//对原图像张量进行降维处理，获取一个通道的量
		auto input_tensorChange = input_tensor.squeeze(0);
		input_tensor = input_tensorChange[2].unsqueeze(0).unsqueeze(0);

		//进行超分处理
		module.to(at::kCPU);
		module.eval();
		at::Tensor result = module.forward({ input_tensor }).toTensor();

		//对输出结果进行处理
		result = result.to(torch::kCPU);
		result = result.squeeze().detach().permute({ 1, 2, 0 });
		result = result.mul(255).clamp(0, 255).to(torch::kU8);

		int height = result.size(0);
		int width = result.size(1);
		int channel = result.size(2);

		//初始化新建图像
		resultImg = Mat(height, width, CV_8UC1);
		std::memcpy((void*)resultImg.data, result.data_ptr(), sizeof(torch::kU8) * result.numel());
		bFlag = true;
		return resultImg;
	}
	catch (const c10::Error & e)
	{
		bFlag = false;
		return resultImg;
	}
}

//输入为一个张量的超分处理函数输出为长宽不等彩色的图-ESRGAN超分处理算法
Mat CSRdeal::ESRGANDealImg(Mat image, torch::jit::script::Module module, bool& bFlag)
{
	cv::Mat resultImg;
	try
	{
		//输入图像转换成张量
		image.convertTo(image, CV_32FC3, 1.0f / 255.f);
		auto input_tensor = torch::from_blob(image.data, { 1, image.rows, image.cols, image.channels() });
		input_tensor = input_tensor.permute({ 0, 3, 1, 2 });

		//进行超分处理
		module.to(at::kCPU);
		module.eval();
		at::Tensor result = module.forward({ input_tensor }).toTensor();

		//对输出结果进行处理
		result = result.to(torch::kCPU);
		result = result.squeeze().detach().permute({ 1, 2, 0 });
		result = result.mul(255).clamp(0, 255).to(torch::kU8);

		int height = result.size(0);
		int width = result.size(1);
		int channel = result.size(2);

		//初始化新建图像
		resultImg = Mat(height, width, CV_8UC3);
		std::memcpy((void*)resultImg.data, result.data_ptr(), sizeof(torch::kU8) * result.numel());
		bFlag = true;
		return resultImg;
	}
	catch (const c10::Error & e)
	{
		bFlag = false;
		return resultImg;
	}
}

//包含模型的图片超分处理函数
Mat CSRdeal::SRModuleDealImg(Mat image, torch::jit::script::Module module, bool& bFlag)
{
	cv::Mat resultImg;
	try
	{
		//输入图像转换成张量
		image.convertTo(image, CV_32FC3, 1.0f / 255.f);
		auto input_tensor = torch::from_blob(image.data, { 1, image.rows, image.cols, image.channels() });
		input_tensor = input_tensor.permute({ 0, 3, 1, 2 });

		//进行超分处理
		module.to(at::kCPU);
		module.eval();
		at::Tensor result = module.forward({ input_tensor }).toTensor();

		//对输出结果进行处理
		result = result.to(torch::kCPU);
		result = result.squeeze().detach().permute({ 1, 2, 0 });
		result = result.mul(255).clamp(0, 255).to(torch::kU8);

		int height = result.size(0);
		int width = result.size(1);
		int channel = result.size(2);

		//初始化新建图像
		resultImg = Mat(height, width, CV_8UC3);
		std::memcpy((void*)resultImg.data, result.data_ptr(), sizeof(torch::kU8) * result.numel());
		bFlag = true;
		return resultImg;
	}
	catch (const c10::Error & e)
	{
		bFlag = false;
		return resultImg;
	}
}

//不带模型的图片超分处理函数
Mat CSRdeal::SRDealImgNoModule(std::string ModelPath, cv::Mat image, bool& bFlag)
{
	torch::jit::script::Module module;
	cv::Mat resultImg;
	try
	{
		module.to(at::kCUDA);
		module = torch::jit::load(ModelPath);
		//assert(module != nullptr);
		image.convertTo(image, CV_32FC3, 1.0f / 255.f);
		auto input_tensor = torch::from_blob(image.data, { 1, image.rows, image.cols, image.channels() });
		at::Tensor outImage = torch::zeros({ 1, image.rows, image.cols, image.channels() });
		input_tensor = input_tensor.permute({ 0, 3, 1, 2 });
		outImage = outImage.permute({ 0, 3, 1, 2 });

		module.eval();
		module.to(at::kCPU);

		at::Tensor result = module.forward({ input_tensor, outImage }).toTensor();

		result = result.squeeze().detach().permute({ 1, 2, 0 });
		result = result.mul(255).clamp(0, 255).to(torch::kU8);
		result = result.to(torch::kCPU);

		int height = result.size(0);
		int width = result.size(1);
		int channel = result.size(2);

		resultImg = cv::Mat(height, width, CV_8UC3);
		std::memcpy((void*)resultImg.data, result.data_ptr(), sizeof(torch::kU8) * result.numel());

		return resultImg;
	}
	catch (const c10::Error& e)
	{
		return resultImg;

	}
}
