import pandas as pd
from bs4 import BeautifulSoup
import requests
import json
import time
import re

url_pokedex = 'https://bulbapedia.bulbagarden.net/wiki/List_of_Pok%C3%A9mon_by_National_Pok%C3%A9dex_number'

def fetch_url(url):
    name = url.split('/')[-1]
    path = 'data/' + name + '.json'
    try:
        with open(path, 'r') as fp:
            html = json.load(fp)
    except:
        print('fetching ' + url)
        resp = requests.get(url)
        html = resp.text
        with open(path, 'w') as fp:
            json.dump(html, fp)
        time.sleep(1)
    return html

############################################################
# scrape pokemon names and ID's from Bulbapedia

soup = BeautifulSoup(fetch_url(url_pokedex))
table = soup.find(id='Generation_I').parent.next_sibling.next_sibling

pokedex = []
for row in table.find_all('tr'):
    pokemon = {
        'Kdex': None,
        'Ndex': None,
        'MS': None,
        'Pokemon': None,
        'Type': []
    }
    pokedex.append(pokemon)

df = pd.DataFrame(pokedex)
df.to_csv('data/pokedex.csv', index=False)
df
