
"""
Helper script to download the HAM10000 dataset.

Instructions:
1. Go to Kaggle: https://www.kaggle.com/kmader/skin-cancer-mnist-ham10000
2. Download the dataset (zip file).
3. Extract it.
4. Organize the images into subfolders based on the CSV metadata or use a pre-organized version.
   Structure should be:
   prediction/
     data/
       train/
         akiec/
         bcc/
         ...
       val/
         akiec/
         bcc/
         ...

   Note: The original dataset has all images in one folder and a CSV.
   You will need to write a script to move them into subfolders if you download the raw version.

   Alternatively, look for "HAM10000 split" on Kaggle for a pre-split version.
"""
print("Please follow the instructions in this file to download and organize the dataset.")
