# https://bechdeltest.com/api/v1/doc - data on bechdel test
# https://www.imdb.com/interfaces/ - imdb data

import requests
import csv
import pandas as pd
import numpy as np
from scipy import stats

# Import and process Bechdel Test Scores

r = requests.get("http://bechdeltest.com/api/v1/getAllMovies").json()

r_dict = {}

for movie in r:
    r_dict[movie['id']] = [movie['year'], movie['title'], movie['imdbid'], movie['rating']]

del movie
    
r_df = pd.DataFrame.from_dict(r_dict, orient = 'index', columns=['year', 'title', 'imdbid', 'rating'])

r_df['tconst'] = 'tt' + r_df['imdbid']

r_df = r_df.rename(index = str, columns = {"rating": "bechdel_rating", "averageRating": "imdb_rating"})

# Add info from IMDB

folder = "" # folder where csv from imdb are stored (see comments at the start of the doc)

title_basics = pd.read_csv(folder + '\\title_basics.tsv', sep='\t')
title_ratings = pd.read_csv(folder + '\\title_ratings.tsv', sep='\t')

r_df = r_df.merge(title_ratings, on = 'tconst', how = 'left')
r_df = r_df.merge(title_basics, on = 'tconst', how = 'left')

del title_basics, title_ratings, r

# Dynamics (rating by decades)

r_df['year'] = r_df['year'].astype(int)
r_df['year'].describe()

r_df['bechdel_rating'] = r_df['bechdel_rating'].astype(int)
r_df['bechdel_rating'].describe()

r_df['decade'] = np.floor(r_df['year']/10)*10

r_df = r_df[r_df['decade'] >= 1920]

r_df['decade'].value_counts()

r_df.groupby(['bechdel_rating'])['decade'].value_counts().to_csv(folder + '\\rating_by_decades.csv')

# Bechdel rating and movie rating

r_df.groupby(['bechdel_rating', 'decade'])['imdb_rating'].mean().to_csv(folder + '\\avg_imdb_rating_by_decades_and_bechdel_rating.csv')

r_df[r_df['decade'] >= 2000].groupby(['bechdel_rating']).agg({'imdb_rating': ['mean', 'std', 'count']})

# r_df[(r_df['decade'] >= 2000) & (r_df['numVotes'] < 30)].shape


stats.ttest_ind(r_df[(r_df['decade'] >= 2000) & (r_df['bechdel_rating'] == 1)]['imdb_rating'], 
                r_df[(r_df['decade'] >= 2000) & (r_df['bechdel_rating'] == 0)]['imdb_rating'],
                nan_policy = 'omit')

stats.ttest_ind(r_df[(r_df['decade'] >= 2000) & (r_df['bechdel_rating'] == 1)]['imdb_rating'], 
                r_df[(r_df['decade'] >= 2000) & (r_df['bechdel_rating'] == 2)]['imdb_rating'],
                nan_policy = 'omit')

stats.ttest_ind(r_df[(r_df['decade'] >= 2000) & (r_df['bechdel_rating'] == 1)]['imdb_rating'], 
                r_df[(r_df['decade'] >= 2000) & (r_df['bechdel_rating'] == 3)]['imdb_rating'],
                nan_policy = 'omit')

stats.ttest_ind(r_df[(r_df['decade'] >= 2000) & (r_df['bechdel_rating'] == 2)]['imdb_rating'], 
                r_df[(r_df['decade'] >= 2000) & (r_df['bechdel_rating'] == 3)]['imdb_rating'],
                nan_policy = 'omit')

stats.ttest_ind(r_df[(r_df['decade'] >= 2000) & (r_df['bechdel_rating'] == 0)]['imdb_rating'], 
                r_df[(r_df['decade'] >= 2000) & (r_df['bechdel_rating'] == 3)]['imdb_rating'],
                nan_policy = 'omit')

# Bechdel and genres

r_df = pd.concat([r_df, r_df['genres'].str.split(",", expand = True)], axis = 1, sort = False)

r_df = r_df.rename(index = str, columns = {0: "genre1", 1: "genre2", 2: "genre3"})

r_df_2000 = r_df[r_df['decade'] >= 2000]

r_df_2000_genres = pd.concat([r_df_2000[['imdbid', 'imdb_rating', 'bechdel_rating', 'genre1']].rename(index = str, 
                                       columns = {'genre1': 'genre'}),
                              r_df_2000[['imdbid', 'imdb_rating', 'bechdel_rating', 'genre2']].rename(index = str, 
                                       columns = {'genre2': 'genre'}),
                              r_df_2000[['imdbid', 'imdb_rating', 'bechdel_rating', 'genre3']].rename(index = str, 
                                       columns = {'genre3': 'genre'})])


r_df_2000_genres.groupby(['genre']).agg({'bechdel_rating': ['mean', 'count'], 'imdb_rating': ['mean', 'count']})

r_df_2000[['bechdel_rating', 'imdb_rating']].corr('pearson')

r_df_2000[['bechdel_rating', 'imdb_rating']].mean()
