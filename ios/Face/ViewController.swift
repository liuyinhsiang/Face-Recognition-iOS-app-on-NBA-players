//
//  ViewController.swift
//  Face
//
//  Created by Kathy Su on 18/5/19.
//  Copyright © 2019 ChihTing Su. All rights reserved.
//

import UIKit
import CoreML
import Vision
import FaceCropper


class ViewController: UIViewController, UIImagePickerControllerDelegate, UINavigationControllerDelegate {

    
    var Pickedimage : UIImage?
    var resultText : String?
    var outputFace: UIImage?
    
    let imagePicker = UIImagePickerController()
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        imagePicker.delegate = self
        imagePicker.sourceType = .photoLibrary
        imagePicker.allowsEditing = true
        
    }
    
    
    
    //Pick Picture
    func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey : Any]) {
        
        if let userPickedimage = info[UIImagePickerController.InfoKey.originalImage] as? UIImage {

            Pickedimage = userPickedimage
            
            let inputImage = Pickedimage
            
            inputImage!.face.crop { result in
                switch result{
                case .success(let faces):
                    self.outputFace = faces[0]
                    guard let ciimage = CIImage(image: self.outputFace!) else{
                        fatalError("Cannot convert UIimage to CIimage")
                    }
                    
                    self.detect(image: ciimage)
                case .notFound:
                    self.resultText = "couldn't find any face"
                case .failure:
                    self.resultText = "couldn't find any face"
                }
            }
            performSegue(withIdentifier: "uploadSegue", sender: self)
        }
        
        imagePicker.dismiss(animated: true, completion: nil)
    }
    
    
    
    //DID NOT USE Crop the face
    func cropFace(image: UIImage){
        
        let inputImage = Pickedimage
        
        inputImage!.face.crop { result in
            switch result{
            case .success(let faces):
                self.outputFace = faces[0]
            case .notFound:
                self.resultText = "couldn't find any face"
            case .failure:
                self.resultText = "couldn't find any face"
            }
        }
    }
    
    
    
    //Apply Machine Learning Model
    func detect(image: CIImage) {
        
        guard let model = try? VNCoreMLModel(for: aligned_nba52_u().model) else {
            fatalError("Loading coreML Model Failed")
        }
        
        let request = VNCoreMLRequest(model: model) {(request, error) in
            guard let results = request.results as? [VNClassificationObservation] else{
                fatalError("Model fail to process image")
            }
            
            //get the most probable result
            guard let firstResult = results.first else { return }
            
            //create the label text contents
            let predClass = "\(firstResult.identifier)"
            let predProb = String(format: "%.02f%", firstResult.confidence * 100)
            
            self.resultText = "\(predClass) \nProbability: \(predProb)%"
        }
        
        let handler = VNImageRequestHandler(ciImage: image)
        
        do{
            try handler.perform([request])
        }
        catch{
            print(error)
        }
    }
    
    
    
    @IBAction func uploadPic(_ sender: Any) {
        present(imagePicker, animated: true, completion: nil)
    }
    
    
    
    @IBAction func startCamera(_ sender: Any) {
        performSegue(withIdentifier: "cameraSegue", sender: self)
    }
    
    
    // Pass select picture to second view controller
    override func prepare(for segue: UIStoryboardSegue, sender: Any?) {
        if segue.destination is SecondViewController{
            let vc = segue.destination as? SecondViewController
            vc?.passImage = Pickedimage
            vc?.passLabel = resultText
        }
    }
    
    
}

