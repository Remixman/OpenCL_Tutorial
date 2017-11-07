__kernel void vecadd(__global const double* a,
                     __global const double* b,
                     __global double* c,
                     const int n)
{
  int gid = get_global_id(0);

  if (gid < n)
    c[gid] = a[gid] + b[gid];
}