package(default_visibility = ["//visibility:private"])

cc_library(
    name = "dynamicuda",
    hdrs = glob([
        "modules/dynamicuda/include/**/*.hpp",
    ]),
    includes = [
        "modules/dynamicuda/include"
    ],
)

cc_library(
    name = "opencv_core",
    srcs = glob([
        "modules/core/src/**/*.cpp",
        "modules/core/include/**/*.h",
    ]) + [
        "custom_hal.hpp",
        "cvconfig.h",
        "opencl_kernels_core.hpp",
        "opencv2/opencv_modules.hpp",
        "version_string.inc",
    ],
    hdrs = glob([
        "modules/core/src/**/*.hpp",
        "modules/core/include/**/*.hpp",
    ]),
    copts = [
        "-D__OPENCV_BUILD",
        "-Iexternal/zlib",
        "-Imodules/dynamicuda/include",
    ],
    includes = [
        ".",
        "modules/core/include",
    ],
    linkopts = ["-pthread"],
    visibility = ["//visibility:public"],
    deps = [
        "//external:zlib",
        ":dynamicuda",
    ],
)

cc_library(
    name = "opencv_contrib",
    srcs = glob([
        "modules/contrib/src/**/*.cpp",
        "modules/contrib/include/**/*.hpp",
    ]),
    hdrs = glob([
        "modules/contrib/include/**/*.hpp",
    ]),
    copts = [
        "-D__OPENCV_BUILD",
    ],
    includes = [
        "modules/contrib/include",
    ],
    linkopts = ["-pthread"],
    visibility = ["//visibility:public"],
    deps = [
        "//external:zlib",
        ":dynamicuda",
        ":opencv_core",
        ":opencv_imgproc",
        ":opencv_features2d",
        ":opencv_objdetect",
        ":opencv_calib3d",
        ":opencv_ml"
    ],
)

genrule(
    name = "cvconfig",
    outs = ["cvconfig.h"],
    cmd = """
cat > $@ <<"EOF"
// IJG JPEG
//#define HAVE_JPEG

// PNG
#define HAVE_PNG

// Compile for 'real' NVIDIA GPU architectures
#define CUDA_ARCH_BIN ""

// NVIDIA GPU features are used
#define CUDA_ARCH_FEATURES ""

// Compile for 'virtual' NVIDIA PTX architectures
#define CUDA_ARCH_PTX ""
EOF"""
)

genrule(
    name = "custom_hal",
    outs = ["custom_hal.hpp"],
    cmd = "touch $@",
)

genrule(
    name = "version_string",
    outs = ["version_string.inc"],
    cmd = "echo '\"OpenCV 2.4.13\"' > $@",
)

genrule(
    name = "opencv_modules",
    outs = ["opencv2/opencv_modules.hpp"],
    cmd = """
        echo '#define HAVE_OPENCV_CORE' >> $@
        echo '#define HAVE_OPENCV_IMGCODECS' >> $@
        echo '#define HAVE_OPENCV_IMGPROC' >> $@
    """,
)

cc_library(
    name = "opencv_imgproc",
    srcs = glob([
        "modules/imgproc/src/**/*.cpp",
        "modules/imgproc/src/**/*.hpp",
        "modules/imgproc/src/**/*.h",
        "modules/imgproc/include/**/*.hpp",
        "modules/imgproc/include/**/*.h",
    ]) + ["opencl_kernels_imgproc.hpp"],
    copts = ["-D__OPENCV_BUILD"],
    includes = [
        ".",
        "modules/core/include",
        "modules/imgproc/include",
    ],
    visibility = ["//visibility:public"],
    deps = [":opencv_core"],
)

genrule(
    name = "opencv_imgproc_kernels",
    outs = ["opencl_kernels_imgproc.hpp"],
    cmd = """
      echo '#include "opencv2/core/ocl.hpp"' > $@
      echo '#include "opencv2/core/ocl_genbase.hpp"' > $@
      echo '#include "opencv2/core/opencl/ocl_defs.hpp"' > $@
    """,
)

cc_library(
    name = "opencv_highgui",
    srcs = glob(
        [
            "modules/highgui/src/**/*.cpp",
            "modules/highgui/src/**/*.hpp",
            "modules/highgui/include/**/*.hpp",
            "modules/highgui/include/**/*.h",
        ],
        exclude = glob([
            "modules/highgui/src/window_carbon.cpp",
            "modules/highgui/src/window_w32.cpp",
            "modules/highgui/src/window_QT.cpp",
            "modules/highgui/src/window_gtk.cpp",
            "modules/highgui/src/cap_*.cpp",
        ]),
    ),
    hdrs = ["modules/highgui/include/opencv2/highgui.hpp"],
    copts = ["-D__OPENCV_BUILD"],
    includes = ["modules/highgui/include"],
    visibility = ["//visibility:public"],
    deps = [
        ":opencv_core",
        ":opencv_imgcodecs"
    ],
)

cc_library(
    name = "opencv_imgcodecs",
    srcs = glob([
        "modules/imgcodecs/src/**/*.cpp",
        "modules/imgcodecs/src/**/*.hpp",
        "modules/imgcodecs/include/**/*.hpp",
        "modules/imgcodecs/include/**/*.h",
    ]),
    copts = [
        "-D__OPENCV_BUILD",
        "-Iexternal/libpng_http",
        "-Iexternal/zlib",
    ],
    includes = [
        "modules/imgcodecs/include",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":opencv_core",
        ":opencv_imgproc",
        "@png_archive//:png",
        "//external:zlib",
    ],
)