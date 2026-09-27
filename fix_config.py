f = open('config.py', encoding='utf-8')
content = f.read()
f.close()
content = content.replace(
    '"ugbs_offices_and_contacts.md",',
    '"ugbs_offices_and_contacts.md",\n    "ugbs_campus_locations.md",'
)
f = open('config.py', 'w', encoding='utf-8')
f.write(content)
f.close()
print('Done')
