.PHONY: netflix
netflix:
	make -C datasets netflix.ratings.csv

.PHONY: imdb
imdb:
	make -C datasets imdb.ratings.csv