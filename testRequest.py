import requests
import json

# r = requests.get('https://api.opendota.com/api/proMatches?less_than_match_id=3702169128')
# object = json.loads(r.text)
# print(len(object))

with open('match_result.json') as json_result:
    results = json.load(json_result)
    print(len(results))
    num_results = 0
    for result in results:
        if result == 0 or result == 1:
            num_results += 1
    print(num_results)

with open('match_detail.json') as json_details:
    details = json.load(json_details)
    print(len(details))
    num_details = 0
    for detail in details:
        if detail:
            num_details += 1
    print(num_details)