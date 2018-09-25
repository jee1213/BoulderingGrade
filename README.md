# Bouldering Grade Classification
# Intro
The goal of this project is to build a multi-class classifier that takes 2D coordinates of holds that makes up a bouldering problem on a standardized wall called the Moonboard, and returns the expected grade of the problem.<br> 
Moonboard is a wall with 11x18 gridpoints, where bouldering holds are screwed into each gridpoints. The x coordinates of the board are described in alphabets ranging from A to K and y coordinates in number, 1 to 18. Thus the position of a hold can be expressed as a combination of an alphabet and a number, e.g. 'D16', which is a hold located at (4, 16) from the left bottom corner of the board.
# Data
The Moonboard problem data is scraped from their webpage, moonboard.com, where depending on two pre-defined hold sets (Moonboard Masters 2017 & Moonboard 2016), 11k and 19k labeled problems are available. 
The scraper is built based upon a code provided by Peter Satterthwaite at Stanford University, and improved such that it can perform the desired function under added security (reCAPTCHA) and change of the behavior in contents deployment of the web page. 
The scraper uses Selenium for remote control, and uses python PhantomJS module to save the page contents that are initiated by Java script. 
# Feature engineering
Each bouldering problem has information on<br> 
-Name<br>
-Grade (difficulty)<br>
-Moves (2D coordinates of the holds)<br>
-Route setter<br>
-Method (restrictions on how to use footholds)<br>
-Number of repeats by other users<br>
that we are going to use.<br> 
The primary information that defines the Grade of the problem comes from the Moves.
We encode the information one-hot style, and add a few more features such as
-length (the number of holds used in the problem<br>
-l, r (the x-coordinate of leftmost / rightmost hold, respectively)<br>
-width (r-l)<br>
-path (minimum distance from the start to the end hold while including all the other holds in the problem)<br>
-std (standard deviation of the path)<br>
which later turns out to be the most important features that defines the grade.
# Code
The data is read and model is trained in ipython notebook MoonBoard_GradePrediction.ipynb. When the model is returned, it is saved as RF_model.sav. Then Grads_Classifier_app/app.py creates a locally-hosted web app, where three html pages are handled via Flask. 

