### Hyperion Hackathon

>  ”With his sister, the Titaness [Theia](https://en.wikipedia.org/wiki/Theia), Hyperion fathered [Helios](https://en.wikipedia.org/wiki/Helios) (Sun), [Selene](https://en.wikipedia.org/wiki/Selene) (Moon) and [Eos](https://en.wikipedia.org/wiki/Eos) (Dawn).”



**TASK**

Build something cool. With flask. And with the data provided...



**DATA**

I've included some data to get you started...

- data/pokemon_go.csv



**DATA DICTIONARY**

| Column             | Description                                                  |
| ------------------ | ------------------------------------------------------------ |
| pokedex_id         | the identifier of a Pokemon (numeric; ranges between 1 and 151) |
| latitude           | coordinates of a sighting (numeric)                          |
| longitude          | coordinates of a sighting (numeric)                          |
| local_time         | exact time of a sighting in format `yyyy-mm-dd'T'hh-mm-ss.ms'Z'` (nominal) |
| close_to_water     | did Pokemon appear close (100m or less) to water (boolean)   |
| city               | the city of a sighting (nominal)                             |
| weather            | weather type during a sighting (Foggy Clear, PartlyCloudy, MostlyCloudy, Overcast, Rain, BreezyandOvercast, LightRain, Drizzle,  BreezyandPartlyCloudy, HeavyRain, BreezyandMostlyCloudy, Breezy, Windy,  WindyandFoggy, Humid, Dry, WindyandPartlyCloudy, DryandMostlyCloudy,  DryandPartlyCloudy, DrizzleandBreezy, LightRainandBreezy,  HumidandPartlyCloudy, HumidandOvercast, RainandWindy) |
| temperature        | temperature in celsius at the location of a sighting (numeric) |
| population_density | the population density per square km of a sighting (numeric) |



**RUBRIC**

You will be judged on the "creativity", "usability" and "interactivity" of your flask app. You must:

- use some type of machine learning algorithm
- push the code up to your "public"/personal GitHub accounts
- have a flask front-end that can accept user input
- augment the data with at least one other data source (think web scraping, google maps api, images?)
- use html/css that's prettier than just a couple of white input boxes
- deploy the model to heroku (or similar)



**TEAMS**

- ☀️ Helios: Anita + Finn + Jordan
- 🌑 Selene: Aman + Brandon + Rittick
- 🌅 Eos: Ashley + Marina + Michael