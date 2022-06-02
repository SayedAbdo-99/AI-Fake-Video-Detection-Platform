# AI-Fake-Video-Detection-Platform
### Description:
A project that works to identify fake videos using deep learning and sentiment analysis. Through, firstly generating a deep learning model capable of classifying fake and real videos using a convolution neural network (CNN) and Stochastic Gradient Descent (SGD) optimizer, that achieved an accuracy of (97.6%). Secondly, generating a sentiment analysis model capable of detecting fake videos based on the comments and emotions, using Naive Bayes machine learning algorithms and natural language processing, that achieved an accuracy of (99%).
### Target:
Produce methods to detect the fake videos to solve many problems such as Criminal issues, fake news and validating the information in the videos.
### Tools and Techniques:
Python, Google Colab, Tensorflow, Deep learning (CNN and SGD Optimization), Machin learning (Naive Bayes, NLP, SVM and KNN), Sentiment Analysis, QT Designer GUI, YouTube Data API, Json, and Anaconda Navigator.
### Abstract
Recently, Deep Learning can generate deep fake videos that can change the face of a target video with a source video where the video of the target person doing or saying things the source person is doing. The wrong use of this technology increases the circulation of these fake videos on social media and stir public opinion. So, we use many new solutions to detect these videos and determine whether the video is real or fake. The first solution: using deep learning and sentiment analysis. Firstly, generating a deep learning model capable of classifying fake and real videos. That is through training model on a large scale of the dataset that contains fake and real videos using a convolution neural network (CNN) and Stochastic Gradient Descent (SGD) optimizer that achieved an accuracy of (97.6%). Secondly, generating a sentiment analysis model capable of detecting fake videos based-on the comments and emotions, by training a model on positive and negative sentences or emotions using Naive Bayes machine learning algorithms and natural language processing, and based on a large scale of the dataset to achieve accuracy of (99%). The second solution: Using RNN and Eye Blanking and Sentiment Analysis, through describes forensic spoofing targeting these fake videos. This reveals the lack of human self-inflicted physiological signals that were not well captured when creating fake videos. Based on a new deep learning model Which consists of a convolutional neural network (CNN) and a recurrent neural network (RNN) to capture normal and temporal phenomena in the eye blinking process. Then uses Sentiment Analysis (SA) where We use an ensemble model that combines the Convolutional Neural Network (CNN) and Long Short-Term Memory (LSTM) models to predict the Sentiment Analysis (SA) of Arabic comments on Twitter Videos. The third solution: Proposing a model to detect the fake videos of the object-based type which are created by deleting or adding an item in the video. These videos are detected by extracting the forensic features from the frames to detect the motion residuals. After that, applying the support vector machine (SVM) classifier to classify these features. The particle swarm optimization is applied to the SVM classifier to minimize the loss of the trained data. The model has validated the copy-move forgeries dataset. The validation mercy is the accuracy and the results show that the proposed model has achieved an accuracy of (83.33%).

## Prototype GUI

<table style="border: none">
    <tr>
        <td width="30%" valign="top"> 
            <h3 style="text-align:center" > Login</h3>
            <img src="https://github.com/SayedAbdo-99/AI-Fake-Video-Detection-Platform/blob/main/GUIPrototype/1-login.png" alt="c" >
        </td>
        <td width="30%" valign="top"> 
            <h3 style="text-align:center" > Registration </h3>
            <img src="https://github.com/SayedAbdo-99/AI-Fake-Video-Detection-Platform/blob/main/GUIPrototype/2-registration.PNG" alt="c" >
        </td>
         <td width="40%" valign="top"> 
            <h3 style="text-align:center" > Video Source</h3>
            <img src="https://github.com/SayedAbdo-99/AI-Fake-Video-Detection-Platform/blob/main/GUIPrototype/3-selection.png" alt="c" >
        </td>
    </tr>
    <tr>
    <td> 
    <h2 style="text-align:center" > Video From Disk</h2> 
    </td> 
    </tr>
    <tr>
        <td width="30%" valign="top"> 
            <h3 style="text-align:center" > Video Path</h3>
            <img src="https://github.com/SayedAbdo-99/AI-Fake-Video-Detection-Platform/blob/main/GUIPrototype/4-disk.png" alt="c" >
        </td>
        <td width="30%" valign="top"> 
            <h3 style="text-align:center" > Video to Frames </h3>
            <img src="https://github.com/SayedAbdo-99/AI-Fake-Video-Detection-Platform/blob/main/GUIPrototype/4-disk-InVideoSelection.png" alt="c" >
        </td>
        <td width="40%" valign="top"> 
            <h3 style="text-align:center" > Test Result </h3>
            <img src="https://github.com/SayedAbdo-99/AI-Fake-Video-Detection-Platform/blob/main/GUIPrototype/4-disk-result.png" alt="c" >
        </td>
    </tr>
    <tr>
     <td>
      <h2 style="text-align:center" > Video From Youtube</h2>
     </td>
    </tr>
    <tr>
        <td width="30%" valign="top"> 
            <h3 style="text-align:center" > Video URL</h3>
            <img src="https://github.com/SayedAbdo-99/AI-Fake-Video-Detection-Platform/blob/main/GUIPrototype/5-youtube.png" alt="c" >
        </td>
        <td width="30%" valign="top"> 
            <h3 style="text-align:center" > Video to Frames && get Comments </h3>
            <img src="https://github.com/SayedAbdo-99/AI-Fake-Video-Detection-Platform/blob/main/GUIPrototype/5-youtube-InVideoURL.png" alt="c" >
        </td>
        <td width="40%" valign="top"> 
            <h3 style="text-align:center" > Test Result </h3>
            <img src="https://github.com/SayedAbdo-99/AI-Fake-Video-Detection-Platform/blob/main/GUIPrototype/5-youtube-result.png" alt="c" >
        </td>
    </tr>
</table>



## UML Diagrams

<table style="border: none">
    <tr>
        <td width="50%" valign="top"> 
            <h3 style="text-align:center" > Component Architecture Diagram</h3>
            <img src="https://github.com/SayedAbdo-99/AI-Fake-Video-Detection-Platform/blob/main/UMLDiagrams/component%20architecture%20diagram.jpg" alt="c" >
        </td>
        <td width="50%" valign="top"> 
            <h3 style="text-align:center" > Sequence Diagram </h3>
            <img src="https://github.com/SayedAbdo-99/AI-Fake-Video-Detection-Platform/blob/main/UMLDiagrams/sequence%20diagram.png" alt="c" >
        </td>
    </tr>
    <tr>
        <td width="50%" valign="top"> 
            <h3 style="text-align:center" > Activity Diagram</h3>
            <img src="https://github.com/SayedAbdo-99/AI-Fake-Video-Detection-Platform/blob/main/UMLDiagrams/activity%20diagram.png" alt="c" >
        </td>
        <td width="50%" valign="top"> 
            <h3 style="text-align:center" > Usecase Diagram </h3>
            <img src="https://github.com/SayedAbdo-99/AI-Fake-Video-Detection-Platform/blob/main/UMLDiagrams/usecase%20diagram.png" alt="c" >
        </td>
    </tr>
</table>
