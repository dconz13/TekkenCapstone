# Tekken 7 Capstone Project
#### By Dillon Connolly
12/23/17


Welcome to my Udacity Machine Learning Engineer nanodegree Capstone project!
Below you will find the required elements to set up and run this project. My Capstone
project creates an agent that is able to play the fighting game Tekken 7 remotely on a
Playstation 4. It uses an implementation of a DDQN network to accomplish this task.

#### Proposal
My proposal is provided in the __proposal.pdf__. For a better experience and to view the gifs provided in the Free-Form Visualization section, I recommend viewing my proposal as the original [Google Docs document](https://docs.google.com/document/d/1_Nx28ke67M_vB9xj3CPgHItSL-JsS-XwrG-8WPnxfl4/edit?usp=sharing).

#### Setup
All Python requirements are provided as a pip freeze in the __requirements.txt__ file.

To run the program follow the steps listed below:
1. Open REM4P application
2. Make sure the REM4P 'configure controller' settings match the ones I provided in the
__key mapping backup.rpi__ file.
3. Connect your PS4 Controller to the computer via USB
4. Launch the PS4 Remote Play application via REM4P
  * Make sure that you put the PS4 Remote Play window in the top left corner of your screen. You might have to play around with the settings in the __Vision__ class in __agent.py__ to make sure that the screen is being captured correctly. To verify the numbers you choose, use the __screen_capture.py__ file.
  * Before hitting Start on the PS4 Remote Play application, hit Settings and change the Resolution to Low (360p). This should make downsizing the images to (84,84) for processing faster.
  * Follow the directions on the PS4 Remote Play application for setting up remote play on your PS4.
5. Launch Tekken 7 using the controller plugged into your computer. You can use it to navigate the PS4 menus as if your computer were a TV.
6. If you are training the agent, go to Offline > Practice mode
7. Select your characters and map
8. Once in practice mode, in the settings set the CPU to Hard difficulty and turn on the option to show the commands on screen. It's cool seeing the agent press buttons and see it fly by on the screen. Medium bots aren't difficult enough to learn against and the harder difficulty bots seem to be easily exploitable.
9. In command prompt, navigate to the directory where the files are stored.
10. Make sure to change the parameters in __agent.py__ based on your test
  * __TRIAL:__ Used as the trial number for naming the files and creating the directory to store them in
  * __TOTAL_TESTS:__ Total amount of tests to do. Tests*Episodes is how long the program will run for.
  * __TOTAL_EPISODES:__ Amount of 60 second episodes to perform per test.
  * __main:__  Be sure to comment play or run depending on if you are training or learning. Also comment import_model(agent) if you are not importing a model.
      * import_model(agent)
      * #run(agent)
      * play(agent)
  * __MEMORY_CAPACITY__ This is how much space the SumTree takes up in RAM. Adjust based on your hardware.
10. Launch the application by typing __python agent.py__
  * After the program initializes it will run for the specified time.
  * __IMPORTANT:__ To stop the program is a bit tricky. You have to keep hitting Ctrl+Alt until the REM4P application frees the mouse and keyboard. Then you can do a keyboard interrupt on the command prompt to cancel the application.

###### Python Requirements:
* bleach==1.5.0
* et-xmlfile==1.0.1
* ez-setup==0.9
* h5py==2.7.1
* html5lib==0.9999999
* jdcal==1.3
* Keras==2.0.8
* Markdown==2.6.9
* mss==3.0.1
* numpy==1.13.3
* olefile==0.44
* opencv-python==3.3.0.10
* openpyxl==2.4.9
* pandas==0.21.0
* Pillow==4.3.0
* protobuf==3.4.0
* pydot==1.2.3
* pydot-ng==1.0.0
* pyparsing==2.2.0
* python-dateutil==2.6.1
* pytz==2017.3
* PyYAML==3.12
* scipy==1.0.0rc1
* six==1.11.0
* tensorflow-gpu==1.3.0
* tensorflow-tensorboard==0.1.8
* virtualenv==15.1.0
* Werkzeug==0.12.2

###### Software Requirements
* REM4P by TMACDEV
* PS4 Remote play
* Tekken 7
* Windows

##### Hardware Requirements
* PS4 with remote play enabled
* Dualshock 4 controller
* micro USB to USB A cable
