//
//  SecondViewController.swift
//  Face
//
//  Created by Kathy Su on 2/6/19.
//  Copyright © 2019 ChihTing Su. All rights reserved.
//

import UIKit



class SecondViewController: UIViewController {

    @IBOutlet weak var pickedImage: UIImageView!
    @IBOutlet weak var resultLabel: UILabel!
    
    var passImage : UIImage!
    var passLabel : String!
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        pickedImage.image = passImage
        resultLabel.text = passLabel

    }

    

    /*
    // MARK: - Navigation

    // In a storyboard-based application, you will often want to do a little preparation before navigation
    override func prepare(for segue: UIStoryboardSegue, sender: Any?) {
        // Get the new view controller using segue.destination.
        // Pass the selected object to the new view controller.
    }
    */

}
