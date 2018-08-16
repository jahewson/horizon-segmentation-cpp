cc_library(
    name = "densecrf",
    srcs = glob([
       "src/*.cpp",
       "src/util.h"
    ]),
    hdrs = glob(["include/*.h"]),
    includes = ["include"],
    visibility = ["//visibility:public"],
    deps = [
       "@eigen_archive//:eigen",
       ":liblbfgs"
   ],
)

cc_library(
    name = "liblbfgs",
    srcs = glob([
       "external/liblbfgs/lib/*.c",
       "external/liblbfgs/lib/*.h"
    ]),
    hdrs = ["external/liblbfgs/include/lbfgs.h"],
    includes = ["include", "external/liblbfgs/include"],
    visibility = ["//visibility:public"],
)