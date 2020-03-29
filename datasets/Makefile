%.gz:
	wget https://datasets.imdbws.com/$@

%.tsv: %.tsv.gz
	gunzip $< 

imdb.ratings.csv: title.basics.tsv title.ratings.tsv netflix-prize-data/README
	python imdb_extract.py

common_movies.csv: imdb.ratings.csv netflix.ratings.csv
	python imdb_netflix_comp.py

netflix-prize-data/README: netflix-prize-data.zip
	unzip netflix-prize-data.zip -d netflix-prize-data
	touch netflix-prize-data/README

netflix.ratings.csv: netflix-prize-data/README
	cat netflix-prize-data/combined_data_1.txt netflix-prize-data/combined_data_2.txt netflix-prize-data/combined_data_3.txt netflix-prize-data/combined_data_4.txt | python netflix_clean_data.py >> netflix.ratings.csv
