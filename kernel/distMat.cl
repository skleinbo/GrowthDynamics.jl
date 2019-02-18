__kernel void addCell2(
    __global float2* X,
    __global int* G,
    __global float* d,
    __global float* b,
    __global uint* mask,
    int N,
    int idx,
    float2 new_pos,
    int new_g,
    float new_b,
    float new_d
){
    X[idx] = new_pos;
    G[idx] = new_g;
    d[idx] = new_d;
    b[idx] = new_b;
    mask[idx] = 1;
}

__kernel void deleteCell(
    __global uint* mask,
    int idx
){
    mask[idx] = 0;
}

__kernel void distMat2(
    __global float* D,
    __global float2* X
){
    int x = get_global_id(0);
    int y = get_global_id(1);

    __local float2 p0;
    p0 = X[y];

    float2 p = X[x] - p0;
    for(int d=0;d<2;d++)
        p[d] = min(pow(p[d],2), pow(1.0f-fabs(p[d]),2)); // PBC
    float tmp = sqrt(p[0]+p[1]);
    D[x+y*get_global_size(0)] = tmp;
}


__kernel void update_distMat2(
    __global float* D,
    __global float2* X,
    __local float* buf,
    int y
){
    int x = get_global_id(0);
    __local float2 p0;
    p0 = X[y];

    float2 p = X[x] - p0;
    for(int d=0;d<2;d++)
        p[d] = min(pow(p[d],2), pow(1.0f-fabs(p[d]),2)); // PBC
    float tmp = sqrt(p[0]+p[1]);

    D[x+y*get_global_size(0)] = tmp;
    D[x*get_global_size(0)+y] = tmp;
}


__kernel void birthRates(
  __global float* B,
  __global float4* D,
  __global uint4* mask,
  int N,
  float sigma2,
  float max_density,
  float dCell
){
  float4 rate = (float4)(0.0f,0.0f,0.0f,0.0f);
  int idx = get_global_id(1);
  for(int j=0;j<N/4;j++) {
      rate +=
        convert_float4(mask[j])*
        // min(1.0f,
        //     exp(-0.5f/sigma2*sign(D[N/4*idx+j]-dCell)*pow(D[N/4*idx+j],2))/
        //     exp(-0.5f*dCell*dCell/sigma2)
        // );
            exp(-0.5f/sigma2*pow(D[N/4*idx+j],2));
  }
  float total = 0.0f;
  for(int j=0;j<4;j++)
    total += rate[j];

  B[idx] = max(1.0f-(total-1.0f)/max_density, 0.0f);
}

__kernel void gaussianWeights(
  __global float* W,
  __global float4* D,
  __global uint4* mask,
  int N,
  float sigma2
){
  float4 rate = (float4)(0.0f,0.0f,0.0f,0.0f);
  int idx = get_global_id(0);
  for(int j=0;j<N/4;j++) {
      rate +=
        convert_float4(mask[j])*
            exp(-0.5f/sigma2*pow(D[N/4*idx+j],2));
  }
  float total = 0.0f;
  for(int j=0;j<4;j++)
    total += rate[j];

  W[idx] = total;
}



__kernel void as_test(__global float4* out){
  uint4 in = (uint4)(1,2,3,4);
  out[0] = convert_float4(in);
  //out[0] = (float4)(1.0f, 2.0f, 3.0f, 4.0f);
}

__kernel void surface(
    __global float4* D,
    __global int4* G,
    __global uint4* mask,
    __global int* out,
    int N,
    int g,
    float sigma
){
    int id = get_global_id(0);
    int4 neighbors = (int4)(0,0,0,0);
    for(int j=0;j<N/4;j++) {
        neighbors += (int)(G[id/4][id%4]==g)*convert_int4(mask[j])*(D[N/4*id + j]<=sigma)*
                        (G[j]!=g);   // evaluates to zero if either sigma>D or mask false
    }
    for(int j=1;j<4;j++){
        neighbors[0] += neighbors[j];
    }
    out[id] = neighbors[0];
}

__kernel void surface_alt(
    __global float* D,
    __global int* G,
    __global uint* mask,
    __global int* out,
    int N,
    int g,
    float sigma
){
    int id = get_global_id(0);
    int neighbors = 0;
    for(int j=0;j<N;j++) {
        neighbors += (int)(G[id]==g)*(D[j+N*id]<=sigma);
                        // (G[j]!=g);   // evaluates to zero if either sigma>D or mask false
    }
    out[id] = neighbors;
}
