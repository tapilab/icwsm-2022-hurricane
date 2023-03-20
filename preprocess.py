import carmen
import gzip
import json
import re
import numpy as np
import pandas as pd
from tqdm import tqdm

def load_tweets(fpath):
    '''Load tweets from a json file.'''
    tweets = []
    users = []
    places = []
    with gzip.open(fpath, 'rb') if fpath.endswith('.gz') else open(fpath, 'rb') as file:
        for line in tqdm(file):
            obj = json.loads(line)
            if 'text' in obj:
                tweets.append(obj)
            else:
                if 'users' in obj:
                    users += obj['users']
                if 'places' in obj:
                    places += obj['places']

    users = {user['id']: user for user in users}
    places = {place['id']: place for place in places}
    return dict(tweets=tweets, users=users, places=places)


def get_center_coords(bbox):
    return [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]

def aggregate_tweet_info(tweet_data):
    '''Add user and place info to tweet dicts, to prepare for geolocating tweets.'''
    tweets = tweet_data['tweets']
    users = tweet_data['users']
    places = tweet_data['places']
    for tweet in tweets:
        tweet['user'] = users[tweet['author_id']]
        if 'geo' in tweet:
            if 'coordinates' in tweet['geo']:
                tweet['coordinates'] = tweet['geo']['coordinates'].copy() # includes 'type', 'coordinates'
            if 'place_id' in tweet['geo']:
                place_info = places.get(tweet['geo']['place_id'])
                if place_info:
                    tweet['coordinates'] = tweet.get('coordinates') or {'coordinates': get_center_coords(place_info['geo']['bbox'])}
                    tweet['place'] = place_info
    return tweets


def geolocate(tweets):
    '''Geolocate tweets using carmen.'''
    resolver = carmen.get_resolver(order=('geocode', 'profile'))
    resolver.load_locations()
    for tweet in tqdm(tweets):
        res = resolver.resolve_tweet(tweet)
        if res:
            location = res[1]
            tweet['location'] = {
                'latitude': location.latitude,
                'longitude': location.longitude,
                'country': location.country,
                'state': location.state,
                'county': location.county,
                'city': location.city
            }


def extract_info(tweet):
    info = {
        'tweet_id': tweet['id'],
        'user_id': tweet['user']['id'],
        'username': tweet['user']['username'],
        'user_location': tweet['user'].get('location'),
        'tweet_time': pd.Timestamp(tweet['created_at']),
        'text': re.sub(r'\r+\n*', '\n', tweet['text']),
        'tweet_place_type': None,
        'tweet_place_name': None,
        'tweet_lat': None,
        'tweet_lon': None,
        'loc_lat': None,
        'loc_lon': None,
        'loc_country': None,
        'loc_state': None,
        'loc_county': None,
        'loc_city': None,
    }
    if info['user_location']:
        info['user_location'] = re.sub(r'\r+', '', info['user_location'])
    
    if 'place' in tweet:
        info['tweet_place_name'] = tweet['place']['full_name']
        info['tweet_place_type'] = tweet['coordinates'].get('type')

    if 'coordinates' in tweet:
        info['tweet_lon'], info['tweet_lat'] = tuple(tweet['coordinates']['coordinates'])
    
    if 'location' in tweet:
        info.update({
            'loc_lat': tweet['location']['latitude'],
            'loc_lon': tweet['location']['longitude'],
            'loc_country': tweet['location']['country'],
            'loc_state': tweet['location']['state'],
            'loc_county': tweet['location']['county'],
            'loc_city': tweet['location']['city']
        })
    return info

def convert_located_tweets_to_dataframe(tweets):
    '''Convert geolocated tweets data from dict to Dataframe.'''
    return pd.DataFrame([extract_info(tweet) for tweet in tweets])


def load_tweets_csv(fpath):
    '''Load geolocated tweets from .csv to dataframe.'''
    return pd.read_csv(fpath, dtype={'tweet_id': str, 'user_id': str},
                       parse_dates=['tweet_time'], infer_datetime_format=True)


def is_none(v, empty_str=False):
    return v is None or v is np.nan or empty_str and v == ''

def tweet_summary(tweets):
    '''Summary of geolocated tweets in dataframe.'''
    print(f"""{len(tweets)} tweets
    {np.sum([not is_none(p) for p in tweets.tweet_place_name])} with original geo info
    {np.sum([not is_none(p) for p in tweets.loc_country])} geolocated
    {np.sum(tweets.loc_country == 'United States')} from US""")
    
    users = tweets.groupby('username').first()
    print(f"""{len(users)} users
    {np.sum([not is_none(p) for p in users.user_location])} have location in profile
    {np.sum([not is_none(p) for p in users.loc_country])} geolocated""")
    
    us_users = users[users.loc_country == 'United States']
    print(f"""{len(us_users)} US users
    {np.sum([is_none(s) for s in us_users.loc_state])} without state info
    {np.sum([is_none(s) for s in us_users.loc_county])} without county info
    {np.sum([is_none(s) for s in us_users.loc_city])} without city info""")