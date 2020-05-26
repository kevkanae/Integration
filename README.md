# Integration
Image Dataset can be found on kaggle : https://www.kaggle.com/iarunava/cell-images-for-detecting-malaria

The basic aim of this task is to

a. Create 2 or 3 Docker Images using dockerfiles, each having its own required libraries

b. Git push your python code to GitHub and make Jenkins pull it to your specified workspace on your local machine

b. Ask Jenkins to recognize the type of code as to whether it is CNN/ANN or RNN based python code and then push it to its respective docker container

c. Make Jenkins fetch the accuracy of the model trained and add more CRP layers or increment/decrement values of the code if accuracy < 90%

d. If container fails then ask Jenkins to make the container pickup from where it left and push updated code to GitHub
