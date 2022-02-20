# ComponentAR

## 🔥 Description:
When the app first launches, you'll be given the choice to scan the device you wish to learn more about. Whether it's a smartphone or a laptop, our machine learning algorithm will detect it. Our machine learning model can currently recognise these two devices because we only have 3D models for 2-3 laptops and mobile phones owing to a lack of time. Following the device's detection, a 3D model of the device will be augmented, allowing the user to see all of the components employed inside it. For object detection, which is the detection of an electronic gadget, we use the yolov5 and yoloX models. We enhance the relevant electrical device in its 3d model once the gadget has been identified so that the user can explore it.
 
We're utilizing web scraping to get information about the device, such as color and features. Yolo models have been converted to Onnx object detection models so that they can be utilized in continuous feed and operated in a C sharp script. This information is retrieved through a QR code on the device that displays information on the electronic components; the benefit of this is that it can display accurate specs even after the device has been modified and updated by a qualified person. Not only that, but the user can also see the provided details about these device components and learn more about them by selecting the option to see more details.

## 👨🏻‍🔬HOW TO USE:
 - When you first open the app, it will present you with two alternatives. You can choose whether to scan a mobile phone or a laptop, and the camera will open when you make your choice.
 - The device can then be scanned with a camera, and if no device is found, our model will run for the default device model.
 - Also, this model may not be compatible with all devices; in that case, we recommend that you try a different device.
 - The device will be detected using the ML algorithm, and a 3D model of the device will be created, after which the device's details will be retrieved via web scraping.
 - Our project then shows a projection of the gadget being opened and the components inside it being revealed.
 - The user can learn more about the gadget by clicking on the various components depicted in the 3D model and reading the        information supplied.
 - To learn more about the electronic components, select Show More Details from the option given.

## 👩🏼‍🔬 How To use the ML model :
After cloning the git Repo
To use the ML model run the comands in terminal 
`cd 'ML Model'`
`pip install -r repuernmenst.txt`
`py detect.py --source <imge address/ 0 for webcam feed> --weights weights.onnx --conf 0.5`

