#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>
#include <dlib/opencv/cv_image.h>
#include <jni_face_align.h>
#include <iostream>
#include <jni.h>
#include <dlib/opencv.h>

using namespace std;
using namespace dlib;

#ifdef __cplusplus
extern "C" {
#endif

// ========================================================
// JNI Mapping Methods
// ========================================================
#define DLIB_ALIGN_JNI_METHOD(METHOD_NAME) \
  Java_com_tzutalin_dlib_PedestrianDet_##METHOD_NAME

JNIEXPORT void JNICALL
    DLIB_ALIGN_JNI_METHOD(jniAlignFace)(JNIEnv* env, jobject thiz, jlong spAddr, jint rec_x, jint rec_y, jint rec_width, jint rec_height,
                                          jlong src, jlong dst) {
  LOG(INFO) << "jniAlignFace";
  cv::Mat* srcImg = (cv::Mat*)src;
  shape_predictor* sp = (shape_predictor*) spAddr;
  array2d<rgb_pixel> img;
  assign_image(img, cv_image<bgr_pixel>(*srcImg));
  rectangle rec(rec_x, rec_y, rec_x + rec_width, rec_y + rec_height);
  full_object_detection shape = (*sp)(img, rec);
  std::vector<full_object_detection> shapes;
  shapes.push_back(shape);
  dlib::array<array2d<rgb_pixel> > face_chips;
  extract_image_chips(img, get_face_chip_details(shapes), face_chips);
  cv::Mat* dstImg = (cv::Mat*)dst;
  cv::Mat tmp_img = toMat(face_chips[0]);
  tmp_img.assignTo((*dstImg),-1);
}

JNIEXPORT jlong JNICALL
    DLIB_ALIGN_JNI_METHOD(jniCreateSp)(JNIEnv* env, jobject thiz,jstring dat) {
  LOG(INFO) << "jniCreateSp";
  shape_predictor* sp = (shape_predictor*) malloc(sizeof(shape_predictor));
  const char* str_dat;
  str_dat = env->GetStringUTFChars(dat, NULL);
  deserialize(str_dat) >> (*sp);
  env->ReleaseStringUTFChars(dat, str_dat);
  return (long)sp;
}



#ifdef __cplusplus
}
#endif
