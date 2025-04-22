import re

text = 'This is **bold text** and this is **another bold phrase**.'

matches = re.findall(r'\*\*(.*?)\*\*', text)

matches_2 = re.findall(r'\*\*(.*)\*\*', text)



print(matches)

print(matches_2)