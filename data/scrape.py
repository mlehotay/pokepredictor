import pandas as pd
from bs4 import BeautifulSoup
import requests
import json
import time
import re

bulbapedia = 'https://bulbapedia.bulbagarden.net'
url_pokedex = bulbapedia + '/wiki/List_of_Pok%C3%A9mon_by_National_Pok%C3%A9dex_number'

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
re_cell = re.compile('t[hd]')

pokedex = {}
for row in table.find_all('tr')[1:-1]:
    cells = row.find_all(re_cell)
    #id = int(cells[1].text.split('#')[1].split('\n')[0])
    try:
        id = int(cells[1].text.split('#')[1].split('\n')[0])
    except:
        continue

    if id not in pokedex:
        name = cells[3].text.strip()
        wiki =  bulbapedia + cells[2].find('a')['href']
        soup2 = BeautifulSoup(fetch_url(wiki))
        image = 'https://' + soup2.find_all('td', colspan='2')[1].find('img')['src'][2:]

        try:
            type1 = cells[4].text.strip()
        except:
            type1 = None
        try:
            type2 = cells[5].text.strip()
        except:
            type2 = None

        pokedex[id] = {
                'pokedex_id': id,
                'name': name,
                'wiki': wiki,
                'img': image,
                'type_1': type1,
                'type_2': type2
        }

df1 = pd.DataFrame.from_dict(pokedex, orient='index')
df1.to_csv('data/pokedex.csv', index=False)
df1.shape
df1.columns

df2 = pd.read_csv('data/pokemon_go.csv')
df2.shape
df2.columns

df3 = df2.merge(df1)
df3.to_csv('data/pokepredictor.csv', index=False)
df3.shape
df3.columns

df3.head()


###################################################
# sandbox
soup = BeautifulSoup(fetch_url(url_pokedex))
table = soup.find(id='Generation_I').parent.next_sibling.next_sibling
row = table.find_all('tr')[25]
cells = row.find_all(re.compile('t[hd]'))
wiki = bulbapedia + cells[2].find('a')['href']
soup2 = BeautifulSoup(fetch_url(wiki))
image = 'https://' + soup2.find_all('td', colspan='2')[1].find('img')['src'][2:]
