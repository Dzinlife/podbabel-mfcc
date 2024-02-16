#import "TorchModule.h"
#import <Libtorch-Lite/Libtorch-Lite.h>

@implementation TorchModule {
 @protected
  torch::jit::mobile::Module _impl;
}

- (nullable instancetype)initWithFileAtPath:(NSString*)filePath {
  self = [super init];
  if (self) {
    try {
      _impl = torch::jit::_load_for_mobile(filePath.UTF8String);
    } catch (const std::exception& exception) {
      NSLog(@"%s", exception.what());
      return nil;
    }
  }
  return self;
}

- (NSArray<NSNumber*>*)predictAudio:(float*)audioBuffer withLength:(int)length withChannels:(int)channels withSampleRate:(int)sampleRate {
  try {
    at::Tensor tensor = torch::from_blob(audioBuffer, {channels, length}, at::kFloat);
    at::Tensor config = torch::full({1}, sampleRate, at::kInt);
    c10::InferenceMode guard;
    auto outputTensor = _impl.forward({tensor, config}).toTensor();
    float* floatBuffer = outputTensor.data_ptr<float>();
    NSMutableArray<NSNumber*>* results = [NSMutableArray new];
    for (int i = 0; i < outputTensor.numel(); i++) {
      [results addObject:@(floatBuffer[i])];
    }
    return results;
  } catch (const std::exception& exception) {
    NSLog(@"%s", exception.what());
    return nil;
  }
}

@end

