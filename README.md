# Mastering-OpenCV-with-Practical-Computer-Vision-Projects

#Project 1

Marker less position estimation. You will first need to caliberate your camera, yet no worries! You merely need to copy
the chess board image and take roughly 20 pictures with it. Then you must provide those image locations to the vector of strings
named fileList. Next, you need to take your own picture of the American flag and replace dstImg.jpg with it(please print the holy flag from pattern.png)
. If all goes well, in markerPose will lie a matrix where the first 3x3 matrix is the rotation of the flag, 
and the next 3x3 matrix is its translation from the origin. You may invert it to find the camera position.

<b> currently there are bugs, advise against usage</b>
