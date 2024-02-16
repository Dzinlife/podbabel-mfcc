//
//  MFCC.swift
//  podbabel-mfcc
//
//  Created by Wz on 2024/2/16.
//  Copyright © 2024 Fall in Life. All rights reserved.
//

import AVFoundation

enum Errors: Error {
    case mfcc(String)
}


func getAudioBuffer(audioFile: AVAudioFile, framesReadCount: Int, channel: Int? = nil) throws -> ([Float], Int)? {
    let frameCount = AVAudioFrameCount(min(Int64(framesReadCount), audioFile.length - audioFile.framePosition))
    
    guard let buffer = AVAudioPCMBuffer(pcmFormat: audioFile.processingFormat, frameCapacity: frameCount) else {
        throw NSError()
    }
    
    try! audioFile.read(into: buffer, frameCount: frameCount)
    
    let channelCount = Int(buffer.format.channelCount)
    
    if let _channel = channel {
        guard (_channel < channelCount) else {
            throw Errors.mfcc("")
        }
        
        let frames = Int(buffer.frameLength)
        let audioBuffer = buffer.floatChannelData![_channel]
        let audioArray = Array(UnsafeBufferPointer(start: audioBuffer, count: frames))
        
        return (audioArray, 1)
    } else {
        var multiChannelAudioData = [[Float]]()
        for channel in 0..<channelCount {
            let frames = Int(buffer.frameLength)
            let audioBuffer = buffer.floatChannelData![channel]
            let audioArray = Array(UnsafeBufferPointer(start: audioBuffer, count: frames))
            multiChannelAudioData.append(audioArray)
        }
        let flattenedAudioData = multiChannelAudioData.flatMap { $0 }
        
        return (flattenedAudioData, channelCount)
    }
}

func reshape(_ array: [NSNumber], itemsPerRow: Int) -> [[NSNumber]] {
    return stride(from: 0, to: array.count, by: itemsPerRow).map {
        Array(array[$0 ..< min($0 + itemsPerRow, array.count)])
    }
}

func calcMFCC (audioData: [Float], channelCount: Int, sampleRate: Int, mfcc_module: TorchModule, n_mfcc: Int = 2) throws -> [[NSNumber]] {
    var data = audioData
    guard let rawResult = mfcc_module.predictAudio(&data, withLength: Int32(audioData.count / channelCount), withChannels: Int32(channelCount), withSampleRate: Int32(sampleRate))
    else {
        throw Errors.mfcc("")
    }
    
    let result = reshape(rawResult, itemsPerRow: n_mfcc)
    
    return result
}

func loadMFCCModule () throws -> TorchModule {
    if let filePath = Bundle.main.path(forResource: "mfcc_model", ofType: "ptl"),
        let module = TorchModule(fileAtPath: filePath) {
        return module
    } else {
        throw Errors.mfcc("Load MFCC module failed")
    }
}

func getAudioMFCC(url: String, onProgress: ((Float) -> Void)? , completion: @escaping (Error?, [[NSNumber]]) -> Void) {
    DispatchQueue.global(qos: .userInitiated).async {
        
//        let startTime = Date()
        
        var mfcc_module: TorchModule
        do {
            mfcc_module = try loadMFCCModule()
        } catch {
            completion(error, [])
            return
        }
        
        let url = URL(fileURLWithPath: url)
        
        let audioFile = try! AVAudioFile(forReading: url)
        
        let mfccWindowSize = 1 << 16
        
        let mfccHopSize = mfccWindowSize / 4
        
        let windowSize = 1 << 22
        
        let windowSizeOffset = mfccWindowSize - mfccHopSize
        
        let numIterations = Int(ceil(Float(audioFile.length) / Float(windowSize)))
        
        var audioDataSegments: [[Float]] = []
        
        var mfccDataResult: [[NSNumber]] = []
        
        let channelCount = audioFile.fileFormat.channelCount
        
        let sampleRate = audioFile.fileFormat.sampleRate
        
        let lock = NSLock()
        let condition = NSCondition()
        
        // Read Queue
        DispatchQueue.global(qos: .userInitiated).async {
           
            for i in 0..<numIterations {
                let from = i * windowSize
                let to = i == numIterations - 1 ? Int(audioFile.length): from + windowSize + windowSizeOffset
                
                if to <= audioFile.length && (to - from) > mfccHopSize {
                
                    audioFile.framePosition = AVAudioFramePosition(from)
                    
                    let frameCount = to - from
                
                    guard let (audioData, _) = try? getAudioBuffer(audioFile: audioFile, framesReadCount: frameCount) else {
                        completion(Errors.mfcc("Read audio buffer error"), [])
                        return
                    }
                    
                    lock.lock()
                    audioDataSegments.append(audioData)
                    condition.signal()
                    lock.unlock()
                    
                }
            }
        }
        
        // Process Queue
        DispatchQueue.global(qos: .userInitiated).async {
                
            for i in 0..<numIterations {
                
                while audioDataSegments.isEmpty {
                    condition.wait()
                }
                
                lock.lock()
                let audioData = audioDataSegments.removeFirst()
                lock.unlock()

                guard let result = try? calcMFCC(audioData: audioData, channelCount: Int(channelCount), sampleRate: Int(sampleRate), mfcc_module: mfcc_module) else {
                    completion(Errors.mfcc("MFCC calculation failed") , [])
                    return
                }
                mfccDataResult.append(contentsOf: result)
                
                onProgress?(Float(i) / Float(numIterations - 1))
                     
                if (i == numIterations - 1) {
                    
                    DispatchQueue.main.async {
//
//                        let endTime = Date()
//                        let timeInterval: Double = endTime.timeIntervalSince(startTime)
//                        print("Time to execute function: \(timeInterval) seconds")
                        
                        completion(nil ,mfccDataResult)
                    }
                   
                }
            }
        }

    }
}

@available(iOS 13.0.0, *)
func getAudioMFCCAsync(url: String) async throws -> [[NSNumber]] {
    let result: [[NSNumber]] = try await withCheckedThrowingContinuation { continuation in
        getAudioMFCC(url: url, onProgress: { progress in
            print("\(String(format: "%.0f", progress * 100))%")
        }) { (err, result) in
            if let err = err {
                continuation.resume(throwing: err)
            } else {
                print(result)
                continuation.resume(returning: result)
            }
            
        }
    }
    return result
}
