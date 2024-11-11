import pandas as pd

data = [
    ['Head Case Designs Officially Licensed LebensArt Butterfly Romance Art Mix Vinyl Sticker Gaming Skin Case Cover Compatible with Xbox Series S Console', 'Games & Accessories'],
    ['Distroller Baby Tiny MIKRONERLITO',' Dolls & Accessories'],
    ['Squish Mallows Lune Loch Ness Monster Clip on ','Stuffed Animals & Plush Toys'],
    ['KeylessOption Remote Key Fob High Security 4btn for Ford (OUCD6000022)','Vehicle Electronics'],
    ['Youth Unisex Core II Glove, Bright Indigo Kaleidoscope, Medium','Boys Accessories'],
    ]
df = pd.DataFrame(data, columns=['title', 'category_name'])

df.to_csv("user_history1.csv",index=False)