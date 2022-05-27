# scatter_plot_to_csv
Given a scatter plot with well defined axes, this script will extract the data from the plot and save it in a csv file.

To run the code, you first need to install the tesseract OCR library
```
!sudo apt install tesseract-ocr
!pip install pytesseract
```

You can run the script by specifying the image path for the ```input``` flag and desired output path to store the csv for the ```output``` flag. 
ex:
```
!python data_from_scatter_plots.py --input /content/img_scatterplot.png --output /content/gen_data.csv 
```
