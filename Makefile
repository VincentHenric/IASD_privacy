.PHONY: netflix
netflix:
	make -C datasets netflix.ratings.csv

.PHONY: imdb
imdb:
	make -C datasets common_movies.csv
