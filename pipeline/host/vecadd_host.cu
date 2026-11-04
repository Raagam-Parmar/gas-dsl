// Host harness: loads vecadd.ptx via the CUDA Driver API (driver JIT-compiles it).
#include <cuda.h>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#define CHECK(call)                                                        \
    do {                                                                   \
        CUresult err = (call);                                             \
        if (err != CUDA_SUCCESS) {                                         \
            const char *msg = nullptr;                                     \
            cuGetErrorString(err, &msg);                                   \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,  \
                    msg ? msg : "unknown");                                \
            exit(1);                                                       \
        }                                                                  \
    } while (0)

static std::string readFile(const char *path) {
    std::ifstream f(path);
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); exit(1); }
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

int main(int argc, char **argv) {
    const char *ptx_path = (argc > 1) ? argv[1] : "vecadd.ptx";
    const unsigned N = 1 << 20;
    const size_t bytes = N * sizeof(float);

    CHECK(cuInit(0));
    CUdevice dev;
    CHECK(cuDeviceGet(&dev, 0));

    char name[128];
    CHECK(cuDeviceGetName(name, sizeof(name), dev));
    printf("Running on: %s\n", name);

    CUcontext ctx;
    CHECK(cuDevicePrimaryCtxRetain(&ctx, dev));
    CHECK(cuCtxSetCurrent(ctx));

    std::string ptx = readFile(ptx_path);
    CUmodule mod;
    CHECK(cuModuleLoadData(&mod, ptx.c_str()));
    CUfunction kernel;
    CHECK(cuModuleGetFunction(&kernel, mod, "vecAdd"));

    std::vector<float> h_a(N), h_b(N), h_c(N);
    for (unsigned i = 0; i < N; ++i) { h_a[i] = float(i); h_b[i] = 2.0f * i; }

    CUdeviceptr d_a, d_b, d_c;
    CHECK(cuMemAlloc(&d_a, bytes));
    CHECK(cuMemAlloc(&d_b, bytes));
    CHECK(cuMemAlloc(&d_c, bytes));
    CHECK(cuMemcpyHtoD(d_a, h_a.data(), bytes));
    CHECK(cuMemcpyHtoD(d_b, h_b.data(), bytes));

    unsigned n = N;
    void *args[] = { &d_a, &d_b, &d_c, &n };
    unsigned block = 256;
    unsigned grid  = (N + block - 1) / block;
    CHECK(cuLaunchKernel(kernel, grid, 1, 1, block, 1, 1, 0, nullptr, args, nullptr));
    CHECK(cuCtxSynchronize());

    CHECK(cuMemcpyDtoH(h_c.data(), d_c, bytes));
    int errors = 0;
    for (unsigned i = 0; i < N; ++i)
        if (h_c[i] != h_a[i] + h_b[i]) ++errors;
    printf(errors ? "FAILED (%d errors)\n" : "PASSED\n", errors);

    cuMemFree(d_a); cuMemFree(d_b); cuMemFree(d_c);
    cuModuleUnload(mod);
    cuDevicePrimaryCtxRelease(dev);
    return errors != 0;
}
