// __opencl_c_images macro needs to be defined ??
// and __IMAGE_SUPPORT__ macro??? idk cuh

const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | 
						  CLK_ADDRESS_CLAMP |
						  CLK_FILTER_LINEAR;

__kernel void test(read_only image2d_t in, write_only image2d_t out){

	int idxX = get_global_id(0);
	int idxY = get_global_id(1);

	uint4 val = read_imageui(in, sampler, (int2)(idxX, idxY));

	uint4 newVal = (uint4)(254,254,254,254) - val;
	write_imageui(out, (int2)(idxX, idxY), newVal);
}
