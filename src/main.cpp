#include <iostream>
#include <string>
#include <fstream>
#include <cmath>
#include <vector>
#include <algorithm>
#include <CL/cl.h>
#include <memory>
#include <unordered_map>
#include <functional>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

#include "internal.h"

cl_context context;
cl_command_queue commandQueue;
cl_device_id device;

std::unordered_map<std::string, cl_program> programs;
std::unordered_map<std::string, cl_kernel> kernels;

std::vector<cl_mem> memories;

size_t globalWorkSize = 256;
size_t localWorkSize = 32;

void cleanup();

#define CHECK_RES(res, txt) \
		if(res != CL_SUCCESS){ \
				std::cerr << txt << "\n Failed with error (" << res << ")\n"; \
				cleanup(); \
				return 1; \
		} \

bool testFun(int x, bool y){

	std::cout << "test - " << x << " : " << y << std::endl;
	return y;
}

enum StageType { KERNEL_STAGE, FUNCTION_STAGE };

struct Stage {
public:
	StageType m_type;

	virtual ~Stage() {}
protected:
	// protected constructor makes it kinda abstract class
	Stage(StageType type) : m_type {type} {}
};

struct KernelStage : public Stage {
public:
	std::string m_kernelName;

	std::vector<size_t> m_paramSizes;
	std::vector<void *> m_ptrs;

	KernelStage(StageType type, std::string name, std::vector<size_t> params, std::vector<void*> ptrs)
		: Stage(type),
		  m_kernelName {name},
		  m_paramSizes {params},
	  	  m_ptrs {ptrs}
	{ }	

};

void cleanup(){
	for(cl_mem mem : memories){
		clReleaseMemObject(mem);
	}

	for (auto iter = kernels.begin(); iter != kernels.end(); ++iter){
		clReleaseKernel(iter->second);
	}
	for (auto iter = programs.begin(); iter != programs.end(); ++iter){
		clReleaseProgram(iter->second);
	}

    clReleaseCommandQueue(commandQueue);
    clReleaseContext(context);
    clReleaseDevice(device); // added in 1.2
}

// not needed but still interesting
/*template <typename Func, typename ... Args>
bool func_api(Func fn, Args&&... args){

	fn(std::forward<Args>(args)...);
}
*/

struct FunctionStage : public Stage {
public:
	std::function<bool()> m_fn;
	
	FunctionStage(StageType type, std::function<bool()> fn) : Stage(type), m_fn {fn} {}

};

int main(int argc, char* argv[]){

	std::vector<std::shared_ptr<Stage>> stages;
    // arg parsing
	
	// init all params at start and fill them out in arg parsing
	int a = 5;
	bool b = true;
	bool shouldShowImgFormats = false;

	// if sobel
	if (true) {
		int BlurKernelSize = 5;
		std::vector<size_t> sizes = { sizeof(BlurKernelSize) };
		std::vector<void*> ptrs = { &BlurKernelSize };

		std::shared_ptr<Stage> ptr ( new KernelStage(KERNEL_STAGE, std::string("sobel.cl"), sizes, ptrs));  
		stages.push_back(std::move(ptr));
	}

	// if function testFun
	if (true) {
		std::function<bool()> fn = [a, b](){ return testFun(a, b); };
		std::shared_ptr<Stage> ptr ( new FunctionStage(FUNCTION_STAGE, fn));
		stages.push_back(std::move(ptr));
	}

	// TODO: either properly init kernelNames here or make setup accept stages
	std::vector<std::string> kernelNames = { std::string("sobel.cl") };
	if (setup(kernelNames, &context, &commandQueue, &device, &programs, &kernels) != true){
		std::cerr << "Failed to set up opencl\n";
		cleanup();
		return 1;
	}

	if(shouldShowImgFormats) {
		if(displayImageFormats(&context) == false){
			return 1;
		}
	}

	//TODO: handle images of arbitrary size correctly
	// set up images
	int x,y,n;
	std::unique_ptr<unsigned char> data (stbi_load("../images/cover.png", &x, &y, &n, 0));
	std::unique_ptr<unsigned char> outData(new unsigned char[x * y * n]);

	int dims[] = {x, y, n};
	std::cout << "image size:" << std::endl;
	std::cout << "x - " << x << ", y - " << y << ", n - " << n << std::endl;
	
	cl_image_format imageFormat;
	imageFormat.image_channel_order = CL_RGBA;
	imageFormat.image_channel_data_type = CL_UNSIGNED_INT8;

	cl_image_desc imageDescriptor;
	imageDescriptor.image_type = CL_MEM_OBJECT_IMAGE2D;
	imageDescriptor.image_width = x;
	imageDescriptor.image_height = y;
	imageDescriptor.image_depth = 0;
	imageDescriptor.image_row_pitch = 0; //0 lets opencl calculate it
	imageDescriptor.image_slice_pitch = 0;
	imageDescriptor.num_mip_levels = 0; // not using mip mapping
	imageDescriptor.num_samples = 0; // must be 0 according to docs, whats the point of it then?
	imageDescriptor.mem_object = nullptr;
									 
	cl_int imageRes = CL_TRUE;
	cl_mem inImgBuf = clCreateImage( context,
				   					 CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, 
									 &imageFormat,
									 &imageDescriptor,
									 data.get(),
									 &imageRes);
	CHECK_RES(imageRes, "Failed to create in image!");
	memories.push_back(inImgBuf);

	cl_mem outImgBuf = clCreateImage( context,
				   					 CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR, 
									 &imageFormat,
									 &imageDescriptor,
									 outData.get(),
									 &imageRes);
	CHECK_RES(imageRes, "Failed to create out image!");
	memories.push_back(outImgBuf);
	
	// work group stuff
	size_t maxLocalWorkSize;
	CHECK_RES( clGetKernelWorkGroupInfo(kernels[kernelNames[0]], device, CL_KERNEL_WORK_GROUP_SIZE,
						   				sizeof(size_t), &maxLocalWorkSize, nullptr),
				"Failed to query work group info from device!");
	

	size_t maxLocalWorkSizeDims[3];
	clGetDeviceInfo(device,
					CL_DEVICE_MAX_WORK_ITEM_SIZES,
					sizeof(maxLocalWorkSizeDims),
					maxLocalWorkSizeDims,
					nullptr);

	//TODO: figure something better out for local work group sizes
	size_t workDims[] = {x, y};
	size_t groupSize[] = {8, 8};

	std::cout << "max work group size - " << maxLocalWorkSize  
			  << "\n max work local work sizee dims - " << dims[0] << ","
			  << dims[1] << "," << dims[2] << std::endl;
	std::cout << "group size :: x - " << groupSize[0] << ", y - " << groupSize[1] << std::endl;

	for (auto& stage : stages){

		// execute each kernel
		if(stage->m_type == KERNEL_STAGE){
			std::shared_ptr<KernelStage> krnlStagePtr = std::static_pointer_cast<KernelStage>(stage);

			std::string krnlName = krnlStagePtr->m_kernelName;

			// assume the first 2 args of each kernel is the in and out images
			CHECK_RES( clSetKernelArg(kernels[krnlName], 0, sizeof(cl_mem), &inImgBuf),
						"Failed to set kernel arg (inimg)!");
			CHECK_RES( clSetKernelArg(kernels[krnlName], 1, sizeof(cl_mem), &outImgBuf),
						"Failed to set kernel arg (outimg)!");

			for(int i = 0; i < krnlStagePtr->m_paramSizes.size(); i++){

				cl_int res = clSetKernelArg(kernels[krnlName], (i+2), krnlStagePtr->m_paramSizes[i], krnlStagePtr->m_ptrs[i]);
				if (res != CL_SUCCESS){
					std::cerr << "Failed to set param " << (i+2) << " for kernel " << krnlName 
								<< std::endl << "Failed with error " << res << std::endl;
					cleanup();
					return 1;
				}
			}

			CHECK_RES( clEnqueueNDRangeKernel(commandQueue, kernels[krnlName], 2, 0, workDims,
											  groupSize, 0, nullptr, nullptr),
						"Failed to enqueeueeue kernel!");
		}
		else if(stage->m_type == FUNCTION_STAGE){
			
			std::shared_ptr<FunctionStage> fnStagePtr = std::static_pointer_cast<FunctionStage>(stage);
			fnStagePtr->m_fn();
		}
	}

	// output reading
	size_t origin[] = {0, 0, 0};
	size_t readRegion[] = {x, y, 1};

	CHECK_RES( clEnqueueReadImage( commandQueue, outImgBuf, CL_TRUE, origin,
						   			readRegion, x * sizeof(unsigned char) * 4, 0, outData.get(),
								   	0, nullptr, nullptr),
				"Failed to enqueue read iamge! out");

	// reads in input image for debug
	CHECK_RES( clEnqueueReadImage( commandQueue, inImgBuf, CL_TRUE, origin,
						   			readRegion, x * sizeof(unsigned char) * 4, 0, data.get(),
								   	0, nullptr, nullptr),
				"Failed to enqueue read iamge! in");

	clFinish(commandQueue);

	const char *outName = "out.png";
	stbi_write_png(outName, x, y, n, outData.get(), n * x); 

	cleanup();
}
