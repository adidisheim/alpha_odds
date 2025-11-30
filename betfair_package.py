import betfair_data as bfd

paths = [
    "data/raw/2025_09_ProAUSGreyhounds.tar",
]

# ============================================================================
# PROCESS THE ARCHIVE (SLOW)
# ============================================================================
# After first run, you can comment out this section and use interactive Python
# or just keep the markets_dict in memory

market_count = 0
update_count = 0

for file in bfd.Files(paths):
    market_count += 1
    for market in file:
        update_count += 1
    print(f"Markets {market_count} Updates {update_count}", end='\r')
print(f"\nMarkets {market_count} Updates {update_count}")

# Store only the FINAL state of each unique market (not every update)
markets_dict = {}  # Key: market_id, Value: Market object (last update)
for file in bfd.Files(paths):
    for market in file:
        # Each iteration updates the same market, so we keep overwriting
        # with the latest state - this gives us the final state of each market
        markets_dict[market.market_id] = market

print(f"\nStored {len(markets_dict)} unique markets (final states)")
print(f"Example: {list(markets_dict.keys())[0] if markets_dict else 'None'}")

# ============================================================================
# TIP: To avoid re-processing, use one of these methods:
# ============================================================================
# 1. Interactive Python: python3 -i asodjas.py
#    (keeps markets_dict in memory after script finishes)
#
# 2. Comment out the processing section above after first run
#    (then just run the inspection code below)
#
# 3. Use Jupyter notebook for interactive work

# ============================================================================
# HOW TO INSPECT A MARKET:
# ============================================================================

# 1. Get a market by its ID
if markets_dict:
    first_market_id = list(markets_dict.keys())[0]
    market = markets_dict[first_market_id]

    print(f"\n{'=' * 60}")
    print(f"INSPECTING MARKET: {first_market_id}")
    print(f"{'=' * 60}")

    # 2. Check the object type
    print(f"\nObject Type: {type(market)}")
    print(f"Is Market object? {isinstance(market, bfd.Market)}")

    # 3. Access market properties
    print(f"\nMarket Properties:")
    print(f"  market_id: {market.market_id}")
    print(f"  market_name: {market.market_name}")
    print(f"  event_name: {market.event_name}")
    print(f"  venue: {market.venue}")
    print(f"  country_code: {market.country_code}")
    print(f"  status: {market.status}")
    print(f"  total_matched: ${market.total_matched:,.2f}")
    print(f"  market_time: {market.market_time}")
    print(f"  number_of_active_runners: {market.number_of_active_runners}")

    # 4. Inspect runners (list of Runner objects)
    print(f"\nRunners ({len(market.runners)} total):")
    for i, runner in enumerate(market.runners[:3], 1):  # Show first 3
        print(f"  Runner {i}:")
        print(f"    Type: {type(runner)}")
        print(f"    Name: {runner.name}")
        print(f"    Selection ID: {runner.selection_id}")
        print(f"    Status: {runner.status}")
        print(f"    Total Matched: ${runner.total_matched:,.2f}")
        if runner.ex:
            print(f"    Has exchange data: Yes")
            if runner.ex.available_to_back:
                best_back = runner.ex.available_to_back[0]
                print(f"    Best Back: {best_back.price} @ {best_back.size}")

    # 5. See all available attributes
    print(f"\nAll Available Attributes:")
    attrs = [attr for attr in dir(market) if not attr.startswith('_') and not callable(getattr(market, attr, None))]
    print(f"  {', '.join(attrs[:10])}...")  # Show first 10