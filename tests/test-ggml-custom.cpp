#include "ggml.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

#define UNUSED(x) (void)(x)

int main(int argc, char **argv) {
    UNUSED(argc);
    UNUSED(argv);

    const int64_t ne[GGML_MAX_DIMS] = { 1, 131072, 65536, 1 };

    // create a tensor with large dimensions
    struct ggml_tensor tensor = {};
    tensor.type = GGML_TYPE_F32;
    tensor.ne[0] = ne[0];
    tensor.ne[1] = ne[1];
    tensor.ne[2] = ne[2];
    tensor.ne[3] = ne[3];
    tensor.nb[0] = sizeof(float);
    tensor.nb[1] = sizeof(float) * ne[0];
    tensor.nb[2] = sizeof(float) * ne[0] * ne[1];
    tensor.nb[3] = sizeof(float) * ne[0] * ne[1] * ne[2];

    // calculate the expected number of rows
    const int64_t expected_nrows = (int64_t)((uint64_t)ne[1] * (uint64_t)ne[2] * (uint64_t)ne[3]);
    const int64_t nrows = ggml_nrows(&tensor);

    // check if the number of rows is calculated correctly
    if (nrows != expected_nrows) {
        printf("ggml_nrows test failed: expected %ld, but got %ld\n", expected_nrows, nrows);
        return 1;
    }

    printf("ggml_nrows test passed.\n");
    return 0;
}
