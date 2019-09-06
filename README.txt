README


Description:




Instructions on running
packages to install:
sklearn, datetime, numpy, pandas, math, numpy, pandas, csv, statistics, maptlotlib, datetime, os




Order to run files

1. Data Cleaning.ipynb
 1. a. Mission tag ranking.ipynb (optional)
 1. b. Maintenance tag ranking.ipynb 
2. create-maint-features
3. fit_logreg.py




Steps: 
1. pull maint2 report 
Pertinent columns: ['Bu/SerNo', 'Maint Level', 'Type Maint Code', 'Action Taken', 'Manhours', 'Rcvd Date', 'Rmvd PartNo', 'Rcvd EOC Code']

2. pull sharp data 
pertinent columns: ['LaunchDate', 'Buno','TFT', 'Land ICAO', 'Pilot Rank']

3. put data into XForce_files/maint_data and XForce_files/sharp_data. 

4. Run 'Data Cleaning.ipynb'
files needed: buno.npy, maintenance2 files, sharp file


After this line is optional. Although the code runs it has not been validated
----------------------------------------------------------------------------------------

5. run 'create-maint-features'
Columns needed: Bu/SerNo, Comp. Date, Action Taken
file needed: maint2 report

6. run 'python fit_logreg.py --save model.pt'
note: put that directly into the terminal

7. run 'python validate_ranking.py --test-bunos.txt --model saved-sessions/model.pt' --date MM/DD/YYYY' (replace MM/DD/YYYY with your own date)
