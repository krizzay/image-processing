// __opencl_c_images macro needs to be defined ??
// and __IMAGE_SUPPORT__ macro??? idk cuh

const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | 
						  CLK_ADDRESS_CLAMP |
						  CLK_FILTER_LINEAR;

__kernel void test(read_only image2d_t in, write_only image2d_t out){

	int idxX = get_global_id(0);
	int idxY = get_global_id(1);

	/*
	if(idxX == 0 && idxY == 0){
		int h = get_image_height(in);
		int w = get_image_width(in);
		printf("h - %d , w - %d \n\n", h, w);

		for(int i = 0; i < 100; i++){

			int ii = i % 10;
			int iii = i / 10;

			printf("i (%d %d)\n", ii, iii);
			uint4 val = read_imageui(in, sampler, (int2)(ii, iii));
			printf("v - %v4d , \n", val);
		}
	}
	*/

	uint4 val = read_imageui(in, sampler, (int2)(idxX, idxY));

	uint4 newVal = (uint4)(254,254,254,254) - val;

//	if(idxX == 6){
//		printf(" id - (%d , %d )\n ", idxX, idxY);
//		printf("val - %v4d \n", val);
//		printf("val new - %v4d \n", newVal);
//	}

	write_imageui(out, (int2)(idxX, idxY), newVal);
}
