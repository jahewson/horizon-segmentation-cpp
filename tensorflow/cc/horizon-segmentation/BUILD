load("//tensorflow:tensorflow.bzl", "tf_cc_binary")

tf_cc_binary(
    name = "horizon-segmentation",
    srcs = ["horizon-segmentation.cc", "horizon-segmentation.h"],
    deps = [
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:client_session",
        "//tensorflow/core:tensorflow",
        "@densecrf//:densecrf",
        "@opencv2//:opencv_core",
        "@opencv2//:opencv_highgui",
        "@opencv2//:opencv_imgproc"
    ]
)
