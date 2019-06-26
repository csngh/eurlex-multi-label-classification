# ATML-Project
ATML Project Eurlex Dataset MultiClass Problem

## Preprocessing:
  Download arff File and eurovoc qrel\
  Put in same Folder as 3 Python files
  1. Execute preprocess.py
  2. call preprocessLabels.py
  3. Use BuildVectors.py to build number vectors

## Scraping and Cleaning:
  
  Note: Install Libraries like *pandas, nltk, bs4 and pathlib* before running the code! 
  
  Download the dataset zip file from: http://www.ke.tu-darmstadt.de/files/resources/eurlex/eurlex_download_EN_NOT.sh.gz
  1. Run the .sh script in terminal/cmd to get all the html files.
  2. Put in same Folder as the .py files
  3. Run data_scraping.py
  4. You'll get a file named "final_scraped.csv" in your folder containing all the scraped text and labels (uncleaned!).
  5. Run data_cleaning.py within the same folder as "final_scraped.csv"
  6. You'll get a file named "final_cleaned.csv" in your folder containing all the cleaned text and labels.
  7. Run remove_labels.py or duplicate_labels.py within the same folder as "final_cleaned.csv" to sort imbalance issues.
  8. You'll get a file named "imabalanced_labeles_removed.csv" or "imbalanced_labels_duplicated.csv" based on your script selection from step 5. 

  *You can find the sorted data in a .csv file here: https://drive.google.com/open?id=1cJQiNfzbkKRwRs8TdrQNQBDPXecb9M4Y*
