import UIKit

class ViewController: UIViewController {
    @IBOutlet var imageView: UIImageView!
    @IBOutlet var resultView: UITextView!

    override func viewDidLoad() {
        super.viewDidLoad()
        
        guard let filePath = Bundle.main.path(forResource: "stereo44100 2", ofType: "mp3") else {
            print("Failed to get the audio file path")
            return
        }

        getAudioMFCC(url: filePath, onProgress: { progress in
            print("\(String(format: "%.0f", progress * 100))%")
        }) { (_, result) in
            print(result)
            print("done, \(result.count)")
            
            let jsonData = try! JSONSerialization.data(withJSONObject: result, options: [])
            let documentDirectory = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
            let fileURL = documentDirectory.appendingPathComponent("output.json")
            try! jsonData.write(to: fileURL)
        }
        
        
    }
}
