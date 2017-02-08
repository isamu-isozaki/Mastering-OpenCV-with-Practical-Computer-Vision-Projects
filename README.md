# Mastering-OpenCV-with-Practical-Computer-Vision-Projects

#Project 1

Marker less position estimation. You will first need to caliberate your camera, yet no worries! You merely need to copy
the chess board image and take roughly 20 pictures with it. Then you must provide those image locations to the vector of strings
named fileList. Next, you need to take your own picture of the American flag and replace dstImg.jpg with it(please print the holy flag from pattern.png)
. If all goes well, in markerPose will lie a matrix where the first 3x3 matrix is the rotation of the flag, 
and the next 3x3 matrix is its translation from the origin. You may invert it to find the camera position.
<b>Currently the Orb matcher appears to not perform properly, sorry if it fails to detect what you trying to detect</b>

#Project 2

Non-rigid facial detection. The algorithm will detect your face and present markers at locations that it detects where your face is present. The training is done by the images provided by the <a href = "https://github.com/StephenMilborrow/muct">MUCT database</a>. Hence it will be required to be downloaded as an addition to this project, I will clarify below.
<b>Currently working on some bugs where it does not save the annotations to an yml file, it will not fail you in execution as it is currently commented out, if you wish to view it, you can uncomment the section in main.cpp where it is written as</b>

```
save_ft<ft_data>("annotation.yml", face_data);//save to yml file
```
<b>To make it work</b>
<ol>
<li>Extract muct-landmark-vl.tar.gz and rename the folder it extracted as landmark</li>
<li>Extract all muct-*-jpg-v1.tar.gz where * is an alphabetic letter into a folder named pictures where the jpg folder encases all the images</li>
<li>Put the landmark folder and the pictures folder in a folder named muct and put the folder named muct in the same location as the NRFD folder</li>
<li>Execute main.cpp</li>
</ol>
