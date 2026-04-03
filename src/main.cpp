#include <iostream>
#include <string>
#include <fstream>
#include <cmath>
#include <vector>
#include <CL/cl.h>

cl_context context;
cl_command_queue commandQueue;
cl_device_id device;

cl_program program;
cl_kernel kernel;

std::vector<cl_mem> memories;

size_t globalWorkSize = 256;
size_t localWorkSize = 32;

#define CHECK_RES(res, txt) \
		if(res != CL_SUCCESS){ \
				std::cerr << txt << "\n Failed with error (" << res << ")\n"; \
				cleanup(); \
				return 1; \
		} \

static std::string readFile(const char* fileName){                                       
    std::fstream f;
    f.open( fileName, std::ios_base::in );
    if(f.is_open() != true){
        std::cerr << "file aint open :<\n";
        return "oops!";
    }
    //assert( f.is_open() );

    std::string res;
    while( !f.eof() ) {
        char c;
        f.get( c );
        res += c;
    }
    
    f.close();

    return std::move(res);
}


// dont know if this is needed
int getLocalWorkSize(size_t maxLocalSize, int globalSize){
    int result = -1;

    for(int i = 1; i <= sqrt(globalSize); i++){
        if (globalSize % i == 0){

            if(globalSize/i == i){
                if(i >= result && i <= maxLocalSize && i < (int)sqrt(globalSize)){
                    result = i;
                }
            }
            else{
                if(i >= result && i <= maxLocalSize && i < (int)sqrt(globalSize)){
                    result = i;
                }
                if(globalSize/i >= result && globalSize/i <= maxLocalSize){
                    result = globalSize/i;
                }
            }
        }
    }

    if(result == -1){
        std::cerr << "couldnt find a factor for " << globalSize << " something strage is ";
    }

    return result;
}               

/*
 Some OpenCL-compliant devices don’t support image processing.
 To check for image support from the host, call clGetDeviceInfo with the 
 CL_DEVICE_IMAGE_SUPPORT option. If the result is CL_FALSE, the device doesn’t 
 support images. On the kernel, the __IMAGE_SUPPORT__ macro will be set to 1 
 if images are supported. If not, the macro will be undefined. 
*/

bool setup(const char* _KernelFileName){
    cl_int platformResult = CL_SUCCESS;
    cl_uint numPlatforms = 0;
    cl_platform_id platforms[64];

    platformResult = clGetPlatformIDs( 64, platforms, &numPlatforms );

    if (platformResult != CL_SUCCESS) {
        std::cerr << "Couldnt get platform IDs!\n Failed with error(" << platformResult << ")\n";
        return false;
    }

    for(int i = 0; i < numPlatforms; i++){
		char version[128];
		clGetPlatformInfo(platforms[i], CL_PLATFORM_VERSION, sizeof(version), version, NULL);
		std::cout << "Platform version: " << version << std::endl;

        cl_device_id devices[64];
        unsigned int deviceCount;
        cl_int deviceResult = clGetDeviceIDs( platforms[i], CL_DEVICE_TYPE_GPU, 64, devices, &deviceCount);
		std::cout << deviceCount << " devices found\n";

        if( deviceResult == CL_SUCCESS){
            for (int j = 0; j < deviceCount; j++){
                unsigned long numComputeUnits;
				bool imageSupport;
                size_t computeUnitsLen;
                cl_int deviceInfoResult = clGetDeviceInfo( devices[j], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &numComputeUnits, &computeUnitsLen);
                cl_int deviceInfoResult2 = clGetDeviceInfo( devices[j], CL_DEVICE_IMAGE_SUPPORT, sizeof(cl_bool), &imageSupport, &computeUnitsLen);
                std::cout << numComputeUnits << " compute units on the device\n";
				std::cout << (imageSupport == CL_TRUE ? "yes" : "no") << " imageSupport on device\n";

                
                if(deviceInfoResult == CL_SUCCESS){
                    // currently picks first device, rank by mem size ig?
                    std::cout << "device found!!\n";
                    device = devices[j];
                    break;
                }
            }

            char version[128];
            clGetDeviceInfo(device, CL_DEVICE_VERSION, sizeof(version), version, NULL);
            printf("OpenCL version: %s\n", version);
            
        }
    }

    cl_int contextResult;
    context = clCreateContext( nullptr, 1, &device, nullptr, nullptr, &contextResult);
    if (contextResult != CL_SUCCESS){
        std::cerr << "Failed to make context!\n Failed with error (" << contextResult << ")\n";
        return false;
    }

    cl_int commandQueueResult;
    cl_queue_properties props[] = { 0 };
    //commandQueue = clCreateCommandQueue(context, device, 0, &commandQueueResult);
    commandQueue = clCreateCommandQueueWithProperties(context, device, props, &commandQueueResult);
    if (commandQueueResult != CL_SUCCESS){
        std::cerr << "Failed to make command queue!\n Failed with error (" << commandQueueResult << ")\n";
        return false;
    }

    // create program and kernel
    std::string s = readFile(_KernelFileName);
    const char* programSource = s.c_str();
    size_t length = 0;
    cl_int programResult;
    program = clCreateProgramWithSource(context, 1, &programSource, &length, &programResult);
    if (programResult != CL_SUCCESS){
        std::cerr << "Failed to make program!\n Failed with error (" << programResult << ")\n";
        return false;
    }

    cl_int programBuildResult = clBuildProgram( program, 1, &device, "-cl-std=CL3.0\0", nullptr, nullptr);
    if (programBuildResult != CL_SUCCESS){
        char log[1024];
        size_t logLength;
        cl_int programBuildInfoResult = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 1024, log, &logLength);

        if(logLength >= 1024){
            char newlog[logLength];
            cl_int programBuildInfoResult = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logLength, newlog, &logLength);

            std::cout << "log len - " << logLength << std::endl;
            std::cout <<  "newlog:\n" << newlog << std::endl << std::endl;
        }else{
            std::cout << "log len - " << logLength << std::endl;
            std::cout <<  "a log:\n" << log << std::endl << std::endl;
        }
        
        if (programBuildInfoResult != CL_SUCCESS){
            std::cerr << "Failed to build program!\n Failed with error (" << programBuildInfoResult << ")\n";
            return false;
        }
    }

    cl_int kernelResult;              // this string must mach entry function name
	kernel = clCreateKernel( program, "roll", &kernelResult);
    if (programResult != CL_SUCCESS){
        std::cerr << "Failed to make kernel!\n Failed with error (" << programResult << ")\n";
        return false;
    }

    return true;
}

void cleanup(){
	for(cl_mem mem : memories){
		clReleaseMemObject(mem);
	}

    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(commandQueue);
    clReleaseContext(context);
    clReleaseDevice(device); // added in 1.2
}


int main(int argc, char* argv[]){

    // arg parsing
		
	if (setup("../src/kernels/test.cl") != true){
		std::cerr << "Failed to set up opencl\n";
		cleanup();
		return 1;
	}

    // do computation!!
	// TODO:
	// read in image 
	// do operations on them :3
	
	uint32_t imgX = 10;
	uint32_t imgY = 10;
	uint32_t pixSize = 4;

	uint8_t inimg[imgX][imgY][pixSize] = {0};
	uint8_t outimg[imgX][imgY][pixSize] = {0};

	cl_image_format imageFormat;
	imageFormat.image_channel_order = CL_RGBA;
	imageFormat.image_channel_data_type = CL_UNSIGNED_INT8;

	cl_image_desc imageDescriptor;
	imageDescriptor.image_type = CL_MEM_OBJECT_IMAGE2D;
	imageDescriptor.image_width = imgX;
	imageDescriptor.image_height = imgY;
	// imageDescriptor.image_depth is for 3d images
	imageDescriptor.image_row_pitch = imgX;
	imageDescriptor.num_mip_levels = 0; // not using mip mapping
	imageDescriptor.num_samples = 0; // must be 0 according to docs, whats the point of it then?
									 
	cl_int imageRes = CL_TRUE;
	cl_mem inImgBuf = clCreateImage( context,
				   					 CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, 
									 &imageFormat,
									 &imageDescriptor,
									 &inimg,
									 &imageRes);
	CHECK_RES(imageRes, "Failed to create in image!");
	memories.push_back(inImgBuf);

	cl_mem outImgBuf = clCreateImage( context,
				   					 CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR, 
									 &imageFormat,
									 &imageDescriptor,
									 &outimg,
									 &imageRes);
	CHECK_RES(imageRes, "Failed to create out image!");
	memories.push_back(outImgBuf);
	
	CHECK_RES( clSetKernelArg(kernel, 0, sizeof(cl_mem), inImgBuf),
				"Failed to set kernel arg (inimg)!");
	CHECK_RES( clSetKernelArg(kernel, 1, sizeof(cl_mem), outImgBuf),
				"Failed to set kernel arg (outimg)!");

	size_t workDims[] = {imgX, imgY};
	size_t groupSize[] = {256, 256}; // arbitrary
	CHECK_RES( clEnqueueNDRangeKernel(commandQueue, kernel, 2, 0, workDims, groupSize, 0, nullptr, nullptr),
				"Failed to enqueeueeue kernel!");

	size_t origin[] = {0, 0, 0};
	size_t readRegion[] = {imgX, imgY, 1};
	CHECK_RES( clEnqueueReadImage( commandQueue, outImgBuf, CL_TRUE, origin,
						   			readRegion, imgX, 0, &outimg, 0, nullptr, nullptr),
				"Failed to enqueue read iamge!");

	clFinish(commandQueue);

	for(int i = 0; i < imgX; i++){
			for(int j = 0; j < imgY; j++){
	
				std::cout << outimg[i][j] << ", ";
			}
			std::cout << std::endl;
	}

	cleanup();

}
