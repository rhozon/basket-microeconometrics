# Market basket analysis with microeconometrics

Bought together is not the same as complementary. On a year of grocery purchases of 2,469
households (The Complete Journey, 84.51°: 155,848 trips, 1.47 million lines, 20.9 million weekly
promotion records), this repository separates what lift measures, tests whether discounts,
displays and mailers move complementary categories, and checks where lift still helps.

Case study page, with an interactive simulator:
<https://rhozon.github.io/portfolio/basket-microeconometrics.html>

## What is in it

| Part | Module | What it answers |
|---|---|---|
| Data | `r/fetch.R`, `bmx/data.py` | Download through the completejourney R package (CC0); baskets, 46 categories, store-week marketing measured without the basket's own purchases |
| Basket analysis | `bmx/mba.py` | Lift, lift within trip-size deciles, co-purchase net of household tastes and trip size |
| Demand | `bmx/demand.py` | Purchase incidence on every category's discount depth, displays and mailers, household fixed effects, clustered errors |
| Recommender | `bmx/rec.py` | Basket completion out of time: popularity, habit, habit + lift, habit + adjusted association |
| Tables | `bmx/cli.py` | Benjamini-Hochberg control over the 2,070 cross effects, key pairs, export for the page |

## Results (from `results/`)

| | |
|---|---|
| Mean lift over 1,035 pairs, raw / within trip-size deciles | 2.53 / 1.27 |
| Top-50 pairs by lift that are top-50 by adjusted association | 22 |
| Own effects surviving the false-discovery control (discount / display / mailer) | 4 / 5 / 2 of 46 |
| Cross effects surviving it | 0 of 2,070 for each |
| 10-point deeper discount on pasta sauce, effect on buying pasta | 0.006 points (95% CI up to 0.11), base 4.1% |
| Basket completion AUC, weeks 40 to 53: habit / + lift / + adjusted association | 0.764 / 0.810 / 0.803 |

## Run it

```bash
Rscript r/fetch.R ~/data/completejourney
export BMX_RAW=~/data/completejourney
pip install -e ".[dev]"
pytest -v
python -m bmx.cli analyse
python -m bmx.cli export site/data.json
```

## Sources

- 84.51°, The Complete Journey 2.0, in the completejourney R package (Boehmke and Mortimer, CC0).
- Gentzkow (2007), American Economic Review; Benjamini and Hochberg (1995), JRSS B; Agrawal and
  Srikant (1994), VLDB.

## License

MIT. Rodrigo Hermont Ozon.
