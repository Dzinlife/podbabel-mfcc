#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

@interface TorchModule : NSObject

- (nullable instancetype)initWithFileAtPath:(NSString*)filePath
    NS_SWIFT_NAME(init(fileAtPath:))NS_DESIGNATED_INITIALIZER;
+ (instancetype)new NS_UNAVAILABLE;
- (instancetype)init NS_UNAVAILABLE;
- (nullable NSArray<NSNumber*>*)predictAudio:(float*)audioBuffer withLength:(int)length withChannels:(int)channels withSampleRate:(int)sampleRate NS_SWIFT_NAME(predictAudio(audioBuffer:withLength:withChannels:withSampleRate:));
@end

NS_ASSUME_NONNULL_END
