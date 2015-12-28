

def getConcepts():
  concepts = [['ACCESSORIES'], \
              ['OFFICE SUPPLIES'],\
              ['OTHER DEPARTMENTS'],\
              ['SEASONAL'],\
              ['BAKERY'],\
              ['HARDWARE', 'PAINT AND ACCESSORIES'],\
              ['AUTOMOTIVE'], \
              ['CELEBRATION'], \
              ['COOK AND DINE','SEAFOOD','LIQUOR,WINE,BEER', 'CANDY, TOBACCO, COOKIES'], \
              ['MEDIA AND GAMING'],\
              ['FROZEN FOODS', 'MEAT - FRESH & FROZEN'],\
              ['CONCEPT STORES'],\
              ['PRE PACKED DELI', 'SERVICE DELI'], \
              ['PETS AND SUPPLIES'],\
              ['BOOKS AND MAGAZINES'],\
              ['IMPULSE MERCHANDISE'], \
              ['HORTICULTURE AND ACCESS','LAWN AND GARDEN','PRODUCE'],\
              ['INFANT APPAREL','INFANT CONSUMABLE HARDLINES'], \
              ['HOME DECOR', 'HOME MANAGEMENT', 'HOUSEHOLD CHEMICALS/SUPP', 'HOUSEHOLD PAPER GOODS','BATH AND SHOWER','LARGE HOUSEHOLD GOODS', 'FURNITURE','SLEEPWEAR/FOUNDATIONS',  'BEDDING'], \
              ['PHARMACY OTC', 'PHARMACY RX'], \
              ['1-HR PHOTO'],\
              ['OPTICAL - FRAMES', 'OPTICAL - LENSES'],\
              ['JEWELRY AND SUNGLASSES'],\
              ['ELECTRONICS', 'PLAYERS AND ELECTRONICS','WIRELESS'],\
              ['BRAS & SHAPEWEAR', 'LADIES SOCKS', 'LADIESWEAR', 'SHEER HOSIERY', 'GIRLS WEAR, 4-6X  AND 7-14','BOYS WEAR','FABRICS AND CRAFTS', 'PLUS AND MATERNITY'],\
              ['SWIMWEAR/OUTERWEAR','SPORTING GOODS'], \
              ['CAMERAS AND SUPPLIES'],\
              ['MENS WEAR', 'MENSWEAR'],\
              ['SHOES'],\
              ['FINANCIAL SERVICES'],\
              ['GROCERY DRY GOODS','DSD GROCERY','DAIRY', 'COMM BREAD'],\
              ['HEALTH AND BEAUTY AIDS','PERSONAL CARE', 'BEAUTY'],
              ['TOYS']];
  return concepts;


def makeConceptsSearchable():
  concepts = getConcepts();
  department_groups = dict();
  for i in range(0, len(concepts)):
    for j in concepts[i]:
      department_groups[j] = i;
  return len(concepts), department_groups;
 

