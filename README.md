# alpha_odds

Commission Rate must be caluclated per MARKET (file)

Fixed effects: 
number next to the name (box)
Venue
Number of active runners


We trade at -4' and we compute momentum and all starting at -15' 
Features: 
Momentum on odds, qty traded (each type), atb (qty), atl (qty), spread.
Rolling volatilities on the features.
Last minute version of all this.


Back lay imbalance
book overround: (implied probabilities across dogs.) (above 100 means illiquid for back qand below 100 for lay)


Static in-out: 
t-\tau:  -15' to -4'
t0: 3'
t+1: 1'

Static in:
t-\tau:  -15' to -4'
t0: 3'


Saved to: /data/projects/punim2039/alpha_odds/data/p/greyhound_au/win_2025_May_6.parquet
Place df is empty, not saved
Saved to: /data/projects/punim2039/alpha_odds/data/p/greyhound_au/mdef_2025_May_6.parquet




Ok the first version of the algo will be: 

1) panel of the first X=3 runners based on total volume at tm1. 
2) for each run aggregate the relevant features for the rest of the field (runners X+1 to N). 
3) predict the in and out of the odds. 