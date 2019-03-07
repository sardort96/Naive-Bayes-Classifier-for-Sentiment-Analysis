# Naive-Bayes-Classifier-for-Sentiment-Analysis
Implementation of the Naive Bayes algorithm to perform sentiment analysis on hotel reviews. The goal of this project is to train the Naive Bayes Classifier and use it as a model to predict the labels of new hotel reviews from the test set.

## Background
Travel - fastest growing sectors
TMCs are trying to provide best search experience
Motivation
Decrease time spent on booking

Criteria when choosing a hotel 
Price
Location/proximity *
Stayed before
Local experience 
Convenience 
Affordability 
Referrals 
Special offers *

Optimizing for Sentiment Analysis 
Binary Multinomial Naive Bayes
Our room was excellent, staff were excellent, and the service was superb. vs. Our room was excellent, staff were, and the service superb.
Dealing with negation 
Although the room was large, it was not really worth the money. vs. Although the room was large, it was not NOT_really NOT_worth NOT_the NOT_money.

Part-of-Speech Tagging
Tag a positive review
Build bigrams
Retrieve adjective-noun pairs for labels 
e.g . renovated building, welcoming staff, wonderful time  

 <br /> <br />
![](images/hardware.png)
<br /> <br />
To test the functionality of our software, we practiced on a single Raspberry Pi connected to two Arduino sensors: Big Sound Sensor, and Water Detection Sensor. The following includes some background information on these sensors as well as how we used them to fulfill the purposes of our project.

![](images/soundsensor.png)

![](images/watersensor.png)
## Sensors
![](images/VeryProfessionalSensor-page-001.jpg)
![](images/BigSoundSensor-page-001.jpg)
![](images/BigSoundSensor-page-002.jpg)
## Web Interface
The user, a security officer for example, has to login to the system to turn it on and start surveilance. On clicking "Sign in", the user is redirected to the dashboard. <br /> <br />
![](images/login.png)
The should now be on the dashboard. Note the 3x3 grid of room numbers on the right. This is meant to represent the 9 different rooms you can be monitoring. Because this is a template project, it is possible to alter the number of rooms without inhibiting the overall process of the software. There are also two user input buttons (on/off and submit). Pressing on/off means the user is ready to start the system. On pressing Submit, the sensors begin listening for sounds or vibrations. <br /> <br />
![](images/warehouse.png)
The system should now be active, and the sensors are now checking for intruders. If noise is made, the sound sensor will activate, and if the water sensor contacts water, the alarm will also activate and notify the security officer. <br /> <br />
![](images/room1.png)
![](images/room5.png)
We use two different sensors to demonstrate that Raspberry Pi's can be placed in different rooms and notify the user from different locations. In our case, Raspberry Pi's would be Internet of Things devices that exploit Fog Computing by running the application locally and sending a notice to the server that is hosted up on the Cloud. 

## Acknowledgements
* Jessen Havill, Professor of Computer Science and Benjamin Barney Chair Director at Denison University

