const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | 
						  CLK_ADDRESS_CLAMP |
						  CLK_FILTER_LINEAR;

// s is only a test
__kernel void start(read_only image2d_t in, write_only image2d_t out, int s){

	int idxX = get_global_id(0);
	int idxY = get_global_id(1);

	uint4 Sx = (uint4)(0);
	uint4 Sy = (uint4)(0);

	Sx += read_imageui(in, sampler, (int2)(idxX-1, idxY+1));
	Sx += 2 * read_imageui(in, sampler, (int2)(idxX-1, idxY));
	Sx += read_imageui(in, sampler, (int2)(idxX-1, idxY-1));

	Sx += read_imageui(in, sampler, (int2)(idxX+1, idxY+1));
	Sx += (-2) * read_imageui(in, sampler, (int2)(idxX+1, idxY));
	Sx += read_imageui(in, sampler, (int2)(idxX+1, idxY-1));

	Sy += read_imageui(in, sampler, (int2)(idxX+1, idxY-1));
	Sy += 2 * read_imageui(in, sampler, (int2)(idxX, idxY-1));
	Sy += read_imageui(in, sampler, (int2)(idxX-1, idxY-1));

	Sy += read_imageui(in, sampler, (int2)(idxX+1, idxY+1));
	Sy += read_imageui(in, sampler, (int2)(idxX, idxY+1));
	Sy += read_imageui(in, sampler, (int2)(idxX-1, idxY+1));

	Sx.w = 1;
	Sy.w = 1;
	float4 Syy = convert_float4( Sy );
	float4 Sxx = convert_float4( Sx );

	// normalise
	Syy = Syy / 4;
	Sxx = Sxx / 4;

	float dotXX = dot(Sxx, Sxx);
	float dotXY = dot(Sxx, Syy);
	float dotYY = dot(Syy, Syy);

	if(idxX == 100 && idxY == 100){
	
		printf("Sx - %v4d\n", Sx);
		printf("Sy - %v4d\n", Sy);
		printf("dotXX %f dotXY %f dotYY %f\n", dotXX, dotXY, dotYY);
	}

	uint4 outPix = (uint4)(convert_uint_sat(dotXX),
						   convert_uint_sat(dotXY),
						   convert_uint_sat(dotYY), 
						   254);

	write_imageui(out, (int2)(idxX, idxY), outPix);
}
