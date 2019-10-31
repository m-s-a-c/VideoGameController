# VideoGameController Using Brain Computer Interface rather than using conventional Keyboard/Joystick.
A human brain instructs the physical movement of body parts using sensory motor neurons.  This work performs multi-class classification for left hand movement, right hand movement  and the resting state. The dataset of the corresponding signals has been taken from  www.physionet.org of 50 subjects. 
EDF_2_CSV.py file converts the EDF files format to CSV file format for the data acquired from www.physionet.org
After converting into CSV format, all 150 files are moved inside Formatted_Files
There are two classification python files:
	1. Bhattacharya_Paper_Implementation_20Sub.py for only 20 subjects 3 recordings each.
	2. Bhattacharya_Paper_Implementation_20Sub.py for all 50 subjects 3 recordings each.
