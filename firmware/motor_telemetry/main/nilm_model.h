#pragma once

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define NILM_MODEL_FEATURE_COUNT 20
#define NILM_MODEL_CLASS_COUNT 4

size_t nilm_model_feature_count(void);
const char *nilm_model_feature_name(size_t index);
int nilm_model_predict(const float features[NILM_MODEL_FEATURE_COUNT],
                       float *confidence);
const char *nilm_model_class_name(int class_id);

#ifdef __cplusplus
}
#endif
