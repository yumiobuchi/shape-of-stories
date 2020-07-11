import os
def main():
	# idliststr = ""
	# with open("bookids.txt","r") as f:
	# 	for line in f:
	# 		idliststr+=","+line.rstrip('\n')
	# os.system("python3 scraper.py -a "+idliststr)	#scrapes Project Gutenberg for 1737 books (same ones used in paper by Regan et al)
	os.system("python3 nbClassify.py")				#applies Naive Bayes classification to extract sentiment from books, preps data for clustering
	# os.system("jupyter notebook clustering.ipynb")	#applies kMeans clustering to generate clusters of texts with similar emotional time series
main()