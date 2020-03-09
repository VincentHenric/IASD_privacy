%.gz:
	wget https://datasets.imdbws.com/$@

%.tsv: %.tsv.gz
	gunzip $< 

imdb_ratings.csv: title.basics.tsv title.ratings.tsv
	python imdb_extract.py
